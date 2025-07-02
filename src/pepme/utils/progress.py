import inspect
from typing import Sequence


class RichProgress:
    import rich.progress

    progress = rich.progress.Progress()
    n_active = 0

    def __init__(self, seq: Sequence, description: str | None = None):
        if description is None:
            description = f"{inspect.stack()[1].filename}"
        self.seq = seq
        self.iterator = None
        self.task_id = RichProgress.progress.add_task(
            description=description, total=len(seq)
        )

    def __iter__(self):
        self.iterator = iter(self.seq)
        RichProgress.progress.start()
        RichProgress.n_active += 1
        return self

    def __next__(self):
        try:
            ret = next(self.iterator)
            RichProgress.progress.advance(self.task_id)
            return ret
        except StopIteration:
            RichProgress.progress.remove_task(self.task_id)
            RichProgress.n_active -= 1
            if RichProgress.n_active == 0:
                RichProgress.progress.stop()
            raise StopIteration
