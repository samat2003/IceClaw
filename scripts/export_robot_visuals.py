"""Export reproducible slide visuals: python scripts/export_robot_visuals.py."""

from pathlib import Path

import plotly.express as px
from PIL import Image, ImageDraw, ImageFont

from iceclaw.science.robot_sim import DEMO_TARGETS, simulate

OUT = Path("docs/assets")
OUT.mkdir(parents=True, exist_ok=True)
result = simulate(list(DEMO_TARGETS))
COLORS = ["#38bdf8", "#fb923c", "#a78bfa"]


def font(size, bold=False):
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    return ImageFont.truetype(f"/usr/share/fonts/truetype/dejavu/{name}", size)


def render(tick, overview=False):
    im = Image.new("RGB", (1920, 1080), "#0b1220")
    d = ImageDraw.Draw(im)

    def text(x, y, value, size=28, fill="#dbeafe", bold=False):
        d.text((x, y), value, font=font(size, bold), fill=fill)

    text(70, 45, "ICECLAW  /  MULTI-ROBOT FOLLOW-UP", 24, "#38bdf8", True)
    text(70, 95, "From screening targets to simulated sampling", 49, bold=True)
    text(
        70,
        170,
        "SYNTHETIC DEMO  •  Rule-based agents  •  No model calls or hardware",
        25,
        "#94a3b8",
    )
    stages = ["SCIENCE: select", "DISPATCH: assign", "NAVIGATION: move", "SAMPLING: mock sample"]
    for i, stage in enumerate(stages):
        x = 70 + i * 455
        d.rounded_rectangle((x, 232, x + 425, 302), radius=12, fill="#17243b")
        text(x + 18, 253, stage, 23, bold=True)
    x0, y0, cell = 115, 385, 27

    def pos(row, col):
        return x0 + col * cell, y0 + row * cell

    for n in range(21):
        d.line((x0, y0 + n * cell, x0 + 20 * cell, y0 + n * cell), fill="#24324a")
        d.line((x0 + n * cell, y0, x0 + n * cell, y0 + 20 * cell), fill="#24324a")
    text(115, 335, "GRID ROUTES  /  one cell per tick", 23, "#94a3b8")
    d.rectangle((x0 - 10, y0 - 10, x0 + 10, y0 + 10), fill="white")
    text(140, 360, "BASE", 18)
    for target in DEMO_TARGETS:
        x, y = pos(target.row, target.col)
        d.ellipse((x - 15, y - 15, x + 15, y + 15), outline="white", width=3)
        text(x + 24, y - 15, target.id, 22, bold=True)
    for i, rover in enumerate(result["frames"][tick]):
        points = [pos(f[i]["row"], f[i]["col"]) for f in result["frames"][: tick + 1]]
        if len(points) > 1:
            d.line(points, fill=COLORS[i], width=6)
        x, y = points[-1]
        d.ellipse((x - 10, y - 10, x + 10, y + 10), fill=COLORS[i])
        text(800, 405 + i * 85, f"ROVER {i + 1}", 26, COLORS[i], True)
        text(1025, 405 + i * 85, f"{rover['moves']} moves  /  {rover['state']}", 26)
    sampled = [
        e for e in result["events"] if e["tick"] <= tick and e["action"] == "mock_sample_no_assay"
    ]
    text(
        800, 340, "MISSION OVERVIEW" if overview else f"SIMULATION TICK {tick}", 25, "#94a3b8", True
    )
    text(800, 700, f"{len(sampled)} / 3 targets mock sampled", 39, bold=True)
    text(800, 770, "Exclusive assignments • bounded travel • action log", 24)
    for i, event in enumerate(sampled):
        text(
            800,
            825 + i * 32,
            f"Tick {event['tick']:02d}: {event['rover']} sampled {event['target']}",
            22,
            "#94a3b8",
        )
    text(
        70,
        998,
        "SIMULATION ONLY — no ice assay, terrain physics, collision avoidance or return trip.",
        25,
        "#fbbf24",
        True,
    )
    return im


render(len(result["frames"]) - 1, overview=True).save(OUT / "robot-simulation-slide.png")
frames = [render(t).resize((960, 540)) for t in range(len(result["frames"]))]
frames[0].save(
    OUT / "robot-simulation.gif",
    save_all=True,
    append_images=frames[1:],
    duration=[160] * (len(frames) - 1) + [2200],
    loop=0,
)
rows = [dict(tick=t, **r) for t, frame in enumerate(result["frames"]) for r in frame]
fig = px.scatter(
    rows,
    x="col",
    y="row",
    color="id",
    animation_frame="tick",
    animation_group="id",
    hover_data=["state", "moves"],
    range_x=[-1, 20],
    range_y=[20, -1],
    title="IceClaw: synthetic robot simulation — mock samples, no ice assay",
)
for target in DEMO_TARGETS:
    fig.add_annotation(x=target.col, y=target.row, text=target.id, showarrow=True)
fig.update_yaxes(scaleanchor="x", scaleratio=1)
fig.write_html(OUT / "robot-simulation.html", include_plotlyjs=True, auto_play=False)
print(f"Exported slide PNG, animated GIF and standalone HTML to {OUT}")
