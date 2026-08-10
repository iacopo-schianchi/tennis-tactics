import matplotlib.pyplot as plt
import matplotlib
import numpy as np

from .line_detection import cluster_segments, fit_line_through_segments

def show_detection_debug(
    image,
    raw_mask,
    hull_mask,
    line_mask,
    lines,
    horiz,
    vert,
    corners,
):
    print('Matplotlib backend: ', matplotlib.get_backend())
    fig, axes = plt.subplots(3, 3, figsize=(18, 14))

    axes[0, 0].imshow(image)
    axes[0, 0].set_title("Original")
    axes[0, 0].axis("off")

    # SAM mask
    axes[0, 1].imshow(raw_mask, cmap="gray")
    axes[0, 1].set_title("SAM mask")
    axes[0, 1].axis("off")

    # convex hull padded
    axes[0, 2].imshow(hull_mask, cmap="gray")
    axes[0, 2].set_title("Hull + padding")
    axes[0, 2].axis("off")

    # filtered line mask
    axes[1, 0].imshow(line_mask, cmap="gray")
    axes[1, 0].set_title("Filtered line mask")
    axes[1, 0].axis("off")

    # all lines
    axes[1, 1].imshow(image)

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line.flatten()
            axes[1, 1].plot(
                [x1, x2],
                [y1, y2],
                linewidth=1
            )

    axes[1, 1].set_title(f"All Hough lines ({len(lines)})")
    axes[1, 1].axis("off")

    # horiz/vert
    axes[1, 2].imshow(image)

    for line in horiz:
        x1, y1, x2, y2 = line
        axes[1, 2].plot(
            [x1, x2],
            [y1, y2],
            linewidth=2
        )

    for line in vert:
        x1, y1, x2, y2 = line
        axes[1, 2].plot(
            [x1, x2],
            [y1, y2],
            linewidth=2
        )

    axes[1, 2].set_title(
        f"Horizontal: {len(horiz)} | Vertical: {len(vert)}"
    )
    axes[1, 2].axis("off")

    # grouped lines
    axes[2, 0].imshow(image)

    v_pairs = sorted(
        [
            ((line[0] + line[2]) / 2, line)
            for line in vert
        ],
        key=lambda x: x[0]
    )

    h_pairs = sorted(
        [
            ((line[1] + line[3]) / 2, line)
            for line in horiz
        ],
        key=lambda x: x[0]
    )

    v_centers = [x[0] for x in v_pairs]
    v_segments = [x[1] for x in v_pairs]

    h_centers = [x[0] for x in h_pairs]
    h_segments = [x[1] for x in h_pairs]

    v_clusters = cluster_segments(
        v_centers,
        v_segments
    )

    h_clusters = cluster_segments(
        h_centers,
        h_segments
    )

    # horiz clusters
    for cluster_id, cluster in enumerate(h_clusters):
        color = plt.cm.tab10(cluster_id % 10)

        for line in cluster:
            x1, y1, x2, y2 = line
            axes[2, 0].plot(
                [x1, x2],
                [y1, y2],
                color=color,
                linewidth=2
            )

    # vert clusters
    for cluster_id, cluster in enumerate(v_clusters):
        color = plt.cm.Set2(cluster_id % 8)

        for line in cluster:
            x1, y1, x2, y2 = line
            axes[2, 0].plot(
                [x1, x2],
                [y1, y2],
                color=color,
                linewidth=2
            )
            
    axes[2, 0].set_title(
        f"Clusters: H={len(h_clusters)}, V={len(v_clusters)}"
    )
    axes[2, 0].axis("off")

    # boundary lines
    axes[2, 1].imshow(image)

    if len(h_clusters) >= 2 and len(v_clusters) >= 2:

        top_cluster = min(
            h_clusters,
            key=lambda c: np.mean(
                [(l[1] + l[3]) / 2 for l in c]
            )
        )

        bottom_cluster = max(
            h_clusters,
            key=lambda c: np.mean(
                [(l[1] + l[3]) / 2 for l in c]
            )
        )

        left_cluster = min(
            v_clusters,
            key=lambda c: np.mean(
                [(l[0] + l[2]) / 2 for l in c]
            )
        )

        right_cluster = max(
            v_clusters,
            key=lambda c: np.mean(
                [(l[0] + l[2]) / 2 for l in c]
            )
        )

        fitted = [
            fit_line_through_segments(top_cluster),
            fit_line_through_segments(bottom_cluster),
            fit_line_through_segments(left_cluster),
            fit_line_through_segments(right_cluster),
        ]

        height, width = image.shape[:2]

        for vx, vy, x0, y0 in fitted:

            if abs(vx) > 1e-6:
                t1 = (0 - x0) / vx
                t2 = (width - x0) / vx

                xa = x0 + t1 * vx
                ya = y0 + t1 * vy

                xb = x0 + t2 * vx
                yb = y0 + t2 * vy

            else:
                xa = xb = x0
                ya = 0
                yb = height

            axes[2, 1].plot(
                [xa, xb],
                [ya, yb],
                linewidth=3
            )

    axes[2, 1].set_title("Fitted boundary lines")
    axes[2, 1].axis("off")

    # corners
    axes[0, 0].set_title("Original")

    axes[0, 2].imshow(image)

    if corners is not None:
        xs = [p[0] for p in corners]
        ys = [p[1] for p in corners]

        axes[0, 2].plot(
            xs + [xs[0]],
            ys + [ys[0]],
            linewidth=3
        )

        axes[0, 2].scatter(
            xs,
            ys,
            s=80
        )

        for i, (x, y) in enumerate(corners):
            axes[0, 2].text(
                x,
                y,
                f"  {i}",
                fontsize=14
            )

        axes[0, 2].set_title(
            f"Final corners: {corners}"
        )

    plt.tight_layout()
    plt.show()