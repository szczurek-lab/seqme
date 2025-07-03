import inspect
from typing import Sequence


class RichProgress:
    import rich.progress

    progress = rich.progress.Progress()
    n_active = 0

    def __init__(
        self, seq: Sequence, description: str | None = None, *, verbose: bool = True
    ):
        self.verbose = verbose
        if description is None:
            description = f"{inspect.stack()[1].filename}"
        self.seq = seq
        self.iterator = None
        if self.verbose:
            self.task_id = RichProgress.progress.add_task(
                description=description, total=len(seq)
            )

    def __iter__(self):
        self.iterator = iter(self.seq)
        if self.verbose:
            RichProgress.progress.start()
            RichProgress.n_active += 1
        return self

    def __next__(self):
        try:
            ret = next(self.iterator)
            if self.verbose:
                RichProgress.progress.update(self.task_id, advance=1, refresh=True)
            return ret
        except StopIteration:
            if self.verbose:
                RichProgress.progress.remove_task(self.task_id)
                RichProgress.n_active -= 1
                if RichProgress.n_active == 0:
                    RichProgress.progress.stop()
            raise StopIteration
