import json
import os

DEFAULT_HISTORY_PATH = "chat_history.json"


class Agent:
    def __init__(self, model, name, system_prompt, history_path=DEFAULT_HISTORY_PATH, max_turns=10, max_new_tokens=512, thinking=False, stateless=False):
        self.model = model
        self.name = name
        self.system_prompt = system_prompt
        self.history_path = history_path
        self.max_turns = max_turns
        self.max_new_tokens = max_new_tokens
        self.thinking = thinking
        self.stateless = stateless

        self.context = None
        self.turns = []
        if not stateless:
            self._load()

    def _read_file(self):
        if not self.history_path or not os.path.exists(self.history_path):
            return {}
        try:
            with open(self.history_path) as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return {}

    def _load(self):
        state = self._read_file().get(self.name, {})
        self.context = state.get("context")
        self.turns = state.get("turns", [])

    def save(self):
        if self.stateless or not self.history_path:
            return
        data = self._read_file()
        data[self.name] = {"context": self.context, "turns": self.turns}
        tmp = self.history_path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp, self.history_path)

    def set_context(self, text):
        self.context = text
        self.save()

    def reset(self, keep_context=False):
        self.turns = []
        if not keep_context:
            self.context = None
        self.save()

    def messages(self, extra=None):
        system = self.system_prompt
        if self.context:
            system += "\n\nContext:\n" + self.context
        turns = self.turns[-2 * self.max_turns:] if self.max_turns else self.turns
        return [{"role": "system", "content": system}, *turns, *(extra or [])]

    def ask(self, prompt):
        user = {"role": "user", "content": prompt}

        reply = self.model.generate(
            self.messages([user]),
            max_new_tokens=self.max_new_tokens,
            thinking=self.thinking,
        )
        if not self.stateless:
            self.turns += [user, {"role": "assistant", "content": reply}]
            self.save()
        return reply