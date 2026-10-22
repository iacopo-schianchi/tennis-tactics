import json
import logging
import re
from decimal import Decimal

from analytics.llm import prompts as P
from analytics.llm.agent import DEFAULT_HISTORY_PATH, Agent

import sqlglot
from sqlglot import exp

log = logging.getLogger("multi_agent")

MAX_QUERIES = 4
MAX_ROUNDS = 2
MAX_SQL_ATTEMPTS = 5
MAX_JSON_RETRIES = 2
MAX_REPORT_CHARS = 6000
MAX_RESULT_CHARS = 9000
MAX_CHECK_CHARS = 4000
MAX_GENERATED_ROWS = 200
MAX_ROWS_SHOWN = 30
VERIFIER_SAMPLE_ROWS = 5
HISTORY_EXCHANGES = 3

ALLOWED_TABLES = {"players", "clips", "points", "events"}
BANNED_NODES = tuple(
    getattr(exp, name) for name in
    ("Insert", "Update", "Delete", "Merge", "Create", "Drop", "Alter", "Command", "Into", "TruncateTable", "Copy", "Set")
    if hasattr(exp, name)
)
BANNED_FUNCTION_PREFIXES = ("pg_", "lo_", "dblink")

def extract_json(text):
    start = text.find("{")
    if start < 0:
        raise ValueError("no JSON object found")
    try:
        obj, _ = json.JSONDecoder().raw_decode(text[start:])
    except json.JSONDecodeError as e:
        raise ValueError(f"invalid JSON: {e}")
    if not isinstance(obj, dict):
        raise ValueError("expected a JSON object")
    return obj


def format_rows(rows, max_rows=MAX_ROWS_SHOWN):
    if not rows: return "(no rows returned)"

    def fmt(value):
        if isinstance(value, (float, Decimal)):
            return f"{round(float(value), 3):g}"
        return "NULL" if value is None else str(value)

    cols = list(rows[0].keys())
    lines = [" | ".join(cols)] + [" | ".join(fmt(row[c]) for c in cols) for row in rows[:max_rows]]
    if len(rows) > max_rows:
        lines.append(f"(showing the first {max_rows} of {len(rows)} rows)")
    return "\n".join(lines)


def ask_json(agent, prompt, validate=None, retries=MAX_JSON_RETRIES):
    current, error = prompt, None
    for _ in range(retries + 1):
        reply = agent.ask(current)
        try:
            obj = extract_json(reply)
            if validate:
                validate(obj)
            return obj
        except ValueError as e:
            error = e
            current = (
                f"{prompt}\n\nYour previous reply was rejected: {e}\n"
                f"Previous reply:\n{reply[:1500]}\nReturn only the corrected JSON object."
            )
    raise ValueError(f"{agent.name} did not return valid JSON: {error}")


def extract_sql(text):
    match = re.search(r"```(?:sql)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    return (match.group(1) if match else text).strip()


def validate_sql(sql):
    sql = sql.strip().rstrip(";").strip()
    if not sql:
        raise ValueError("empty query")
    try:
        statements = [s for s in sqlglot.parse(sql, read="postgres") if s is not None]
    except sqlglot.errors.SqlglotError as e:
        raise ValueError(f"SQL syntax error: {str(e)[:300]}")
    if len(statements) != 1:
        raise ValueError("only a single statement is allowed")

    tree = statements[0]
    if not isinstance(tree, exp.Query):
        raise ValueError(f"the query must be a SELECT, not {type(tree).__name__}")
    bad = tree.find(*BANNED_NODES)
    if bad is not None:
        raise ValueError(f"not allowed in a read-only query: {type(bad).__name__}")
    for fn in tree.find_all(exp.Anonymous):
        if fn.name.lower().startswith(BANNED_FUNCTION_PREFIXES):
            raise ValueError(f"function not allowed: {fn.name}")

    cte_names = {cte.alias for cte in tree.find_all(exp.CTE)}
    for table in tree.find_all(exp.Table):
        if not isinstance(table.this, exp.Identifier):
            continue
        if table.db or (table.name not in ALLOWED_TABLES and table.name not in cte_names):
            raise ValueError(f"unknown table: {table.sql(dialect='postgres')}. Use only {sorted(ALLOWED_TABLES)}")

    params = {p.name for p in tree.find_all(exp.Placeholder)}
    if params != {"player_id"}:
        raise ValueError("the query must use %(player_id)s and no other parameters")
    return sql


def render_results(results, budget):
    if not results:
        return "(none)"
    per_result = max(500, budget // len(results))
    blocks = []
    for result in results:
        text = result["text"]
        if len(text) > per_result:
            text = text[:per_result] + "\n[truncated]"
        blocks.append(f"### {result['title']}\n{text}")
    return "\n\n".join(blocks)


def plan_schema(registry_names):
    return {"type": "object", "properties": {
        "action": {"type": "string", "enum": ["report_only", "fetch", "cannot_answer"]},
        "reason": {"type": "string"},
        "queries": {"type": "array", "maxItems": MAX_QUERIES, "items": {"type": "object", "properties": {
            "source": {"type": "string", "enum": ["registry", "new"]},
            "name": {"type": "string", "enum": list(registry_names)},
            "description": {"type": "string"}}, "required": ["source"]}}},
        "required": ["action", "reason", "queries"]}

VERDICT_SCHEMA = {"type": "object", "properties": {
    "correct": {"type": "boolean"}, "explanation": {"type": "string"}},
    "required": ["correct", "explanation"]}

SUFFICIENCY_SCHEMA = {"type": "object", "properties": {
    "sufficient": {"type": "boolean"}, "missing": {"type": "string"}},
    "required": ["sufficient", "missing"]}


class MultiAgent:
    def __init__(self, model, run_sql, queries, schema, player_id, report=None, history_path=DEFAULT_HISTORY_PATH, condensed=()):
        self.run_sql = run_sql
        self.queries = dict(queries)
        self.schema = schema
        self.player_id = player_id
        self.condensed = set(condensed)
        self.last_trace = None

        # agents
        self.planner = Agent(model, "planner", P.PLANNER_SYS, stateless=True, thinking="MEDIUM", max_new_tokens=1024, json_schema=plan_schema(self.queries))
        self.sql_gen = Agent(model, "sql_generator", P.SQL_GEN_SYS, stateless=True, thinking="HIGH", max_new_tokens=2048)
        self.verifier = Agent(model, "verifier", P.VERIFIER_SYS, stateless=True, max_new_tokens=384, json_schema=VERDICT_SCHEMA)
        self.sufficiency = Agent(model, "sufficiency", P.SUFFICIENCY_SYS, stateless=True, max_new_tokens=256, json_schema=SUFFICIENCY_SCHEMA)
        self.synth = Agent(model, "synthesizer", P.SYNTH_SYS, history_path=history_path, max_new_tokens=768)

        sql_context = f"Schema:\n{schema}\n\nConventions:\n{P.DOMAIN_NOTES}"
        self.sql_gen.set_context(sql_context)
        self.verifier.set_context(sql_context)

        if report is not None:
            self.synth.set_context(report)
        self._refresh_planner_context()

    def set_report(self, report):
        self.synth.set_context(report)
        self._refresh_planner_context()

    def reset(self):
        self.synth.reset()
        self._refresh_planner_context()

    def _refresh_planner_context(self):
        listing = "\n".join(f"- {name}: {q.description}" for name, q in self.queries.items())
        report = (self.synth.context or "(no report yet)")[:MAX_REPORT_CHARS]
        self.planner.set_context(
            f"Schema:\n{self.schema}\n\nRegistry queries:\n{listing}\n\nExisting report:\n{report}"
        )

    @property
    def _params(self):
        return {"player_id": self.player_id}

    def query(self, question):
        trace = {"question": question, "plans": [], "sufficiency": [], "results": [], "failures": []}
        self.last_trace = trace
        results, failures, fetched = trace["results"], trace["failures"], set()
        missing = None

        for round_no in range(1, MAX_ROUNDS + 1):
            plan = self._plan(question, results, failures, missing)
            trace["plans"].append(plan)
            log.info("round %d plan: %s", round_no, plan)

            if plan["action"] == "cannot_answer" and not results:
                answer = f"I can't answer this with the available data. {plan.get('reason', '')}".strip()
                self.synth.record(question, answer)
                return answer
            if plan["action"] != "fetch":
                break

            if not self._execute_plan(plan, fetched, results, failures):
                break  # everything proposed was already fetched
            if round_no == MAX_ROUNDS:
                break

            verdict = self._check_sufficient(question, results, failures)
            trace["sufficiency"].append(verdict)
            log.info("sufficiency: %s", verdict)
            if verdict["sufficient"]:
                break
            missing = verdict["missing"]

        return self._synthesize(question, trace["plans"], results, failures)

    # planner

    def _history_text(self):
        turns = self.synth.turns[-2 * HISTORY_EXCHANGES:]
        return "\n".join(f"{t['role']}: {t['content'][:600]}" for t in turns)

    def _plan(self, question, results, failures, missing):
        prompt = f"Question: {question}"
        history = self._history_text()
        if history:
            prompt += f"\n\nRecent conversation (the question may be a follow-up):\n{history}"
        if results or failures:
            done = [r["title"] for r in results] + [f"{f['title']} (failed)" for f in failures]
            prompt += "\n\nAlready fetched:\n" + "\n".join(f"- {d}" for d in done)
        if missing:
            prompt += f"\n\nThe fetched data was not enough. Missing: {missing}"

        def validate(plan):
            action = plan.get("action")
            if action not in ("report_only", "fetch", "cannot_answer"):
                raise ValueError("action must be report_only, fetch or cannot_answer")
            if action == "fetch":
                items = plan.get("queries")
                if not isinstance(items, list) or not items:
                    raise ValueError("fetch needs a non-empty queries list")
                for item in items:
                    if not isinstance(item, dict):
                        raise ValueError("each query must be an object")
                    if item.get("source") == "registry":
                        if item.get("name") not in self.queries:
                            raise ValueError(f"unknown registry query: {item.get('name')}")
                    elif item.get("source") == "new":
                        if not str(item.get("description", "")).strip():
                            raise ValueError("a new query needs a description")
                    else:
                        raise ValueError("source must be registry or new")

        try:
            return ask_json(self.planner, prompt, validate)
        except ValueError as e:
            log.warning("planner failed (%s), answering from the report", e)
            return {"action": "report_only", "reason": "planner failed", "queries": []}

    # execution

    def _execute_plan(self, plan, fetched, results, failures):
        attempted = 0
        for item in plan["queries"][:MAX_QUERIES]:
            if item["source"] == "registry":
                key, title = ("registry", item["name"]), item["name"]
            else:
                title = item["description"].strip()
                key = ("new", title.lower())
            if key in fetched:
                continue
            fetched.add(key)
            attempted += 1

            try:
                if item["source"] == "registry":
                    results.append(self._run_registry(item["name"]))
                else:
                    results.append(self._run_new(title))
            except Exception as e:
                log.warning("query %s failed: %s", title, e)
                failures.append({"title": title, "error": str(e)})
        return attempted

    def _run_registry(self, name):
        query = self.queries[name]
        rows = self.run_sql(query.sql, self._params)
        if name in self.condensed:
            text = query.generate_insight(rows, condensed=True)
        else:
            text = query.generate_insight(rows)
        return {"title": name, "source": "registry", "text": text}

    def _run_new(self, description):
        sql, rows, feedback, previous = None, None, None, None
        for attempt in range(1, MAX_SQL_ATTEMPTS + 1):
            prompt = f"Data needed: {description}"
            if previous:
                prompt += f"\n\nPrevious query:\n{previous}\n\nProblem: {feedback}\nWrite a corrected query."

            sql = extract_sql(self.sql_gen.ask(prompt))
            previous = sql
            try:
                sql = validate_sql(sql)
                self.run_sql("EXPLAIN " + sql, self._params)
                rows = self.run_sql(sql, self._params, max_rows=MAX_GENERATED_ROWS)
            except Exception as e:
                feedback = f"The query failed: {e}"
                log.info("sql attempt %d failed: %s", attempt, e)
                continue

            verdict = self._verify(description, sql, rows)
            if verdict["correct"]:
                return {"title": description, "source": "generated", "sql": sql, "text": format_rows(rows)}
            feedback = f"A reviewer found it incorrect: {verdict['explanation']}"
            log.info("sql attempt %d rejected: %s", attempt, feedback)

        raise RuntimeError(f"no valid query after {MAX_SQL_ATTEMPTS} attempts. Last problem: {feedback}")

    # check SQL correctness
    def _verify(self, description, sql, rows):
        prompt = f"Requested data: {description}\n\nQuery:\n{sql}\n\nResult: {len(rows)} rows. First rows:\n{format_rows(rows, VERIFIER_SAMPLE_ROWS)}"

        def validate(obj):
            if not isinstance(obj.get("correct"), bool): raise ValueError("correct must be true or false")

        try:
            verdict = ask_json(self.verifier, prompt, validate)
        except ValueError as e:
            log.warning("verifier failed (%s), accepting the query", e)
            return {"correct": True, "explanation": ""}
        verdict.setdefault("explanation", "")
        return verdict

    # sufficiency and synthesis

    def _check_sufficient(self, question, results, failures):
        prompt = f"Question: {question}\n\nFetched data:\n{render_results(results, MAX_CHECK_CHARS)}"
        if failures: prompt += "\n\nCould not be fetched:\n" + "\n".join(f"- {f['title']}" for f in failures)

        def validate(obj):
            if not isinstance(obj.get("sufficient"), bool): raise ValueError("sufficient must be true or false")

        try:
            verdict = ask_json(self.sufficiency, prompt, validate)
        except ValueError:
            return {"sufficient": True, "missing": ""}
        verdict["missing"] = str(verdict.get("missing", ""))
        return verdict

    def _synthesize(self, question, plans, results, failures):
        prompt = f"Question: {question}"
        reasons = [p.get("reason", "") for p in plans if p.get("reason")]
        if reasons:
            prompt += "\n\nPlan: " + " ".join(reasons)
        prompt += f"\n\nFetched data:\n{render_results(results, MAX_RESULT_CHARS)}"
        if failures:
            prompt += "\n\nNot available:\n" + "\n".join(f"- {f['title']}: {f['error'][:200]}" for f in failures)

        # q/a in stored history
        return self.synth.ask(prompt, store=question)