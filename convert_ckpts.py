from pathlib import Path

from EMber import EMberModel

root = Path("checkpoints/published-checkpoints")

for checkpoint in sorted(root.glob("*/teacher.*.pth")):
    name = f"EMber-{checkpoint.parent.name}"
    output = Path("checkpoints/hub") / name

    model = EMberModel.from_teacher_checkpoint(checkpoint)
    model.save_pretrained(output)
    print(f"Prepared {output}")
    del model
