"""
Progress display utilities with spinners and progress bars.
Использует библиотеку rich для красивого CLI вывода.
"""

from typing import Optional
from datetime import datetime
from rich.console import Console
from rich.status import Status
from rich.progress import Progress, BarColumn, TaskProgressColumn, TextColumn, TimeRemainingColumn


class ProgressDisplay:
    """Менеджер прогресс-отображения с поддержкой спиннеров и баров."""

    def __init__(self):
        self._console = Console()
        self._status: Optional[Status] = None
        self._spinner_message: str = ""
        self._progress: Optional[Progress] = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _safe_print(self, text: str) -> None:
        """Печатает текст безопасно — с учётом активного Status/Progress."""
        if self._status is not None:
            # Status.console.print() корректно паркует спиннер, печатает строку
            # выше него и возобновляет анимацию
            self._status.console.print(text)
        elif self._progress is not None:
            self._progress.console.print(text)
        else:
            self._console.print(text)

    def _fresh_console(self) -> Console:
        """Создаёт чистую консоль без унаследованного состояния Live."""
        return Console()

    # ------------------------------------------------------------------
    # Spinner  (используем Status — специализированный API Rich для спиннеров)
    # ------------------------------------------------------------------

    def show_spinner(self, message: str = "Research in progress") -> None:
        """Показать спиннер с сообщением."""
        self._spinner_message = message

        if self._status is not None:
            # Просто меняем текст у уже крутящегося спиннера
            self._status.update(message)
            return

        # Каждый раз создаём новую консоль — иначе после Progress.stop()
        # остаётся грязное состояние _live и новый Status не рендерится
        self._console = self._fresh_console()
        self._status = self._console.status(
            f"[blue]{message}[/blue]",
            spinner="dots",
            spinner_style="blue",
        )
        self._status.start()

    def stop_spinner(self) -> None:
        """Остановить спиннер и оставить статичную строку в выводе."""
        if self._status is None:
            return

        # Печатаем финальную строку ДО остановки — она останется в терминале,
        # после чего Status сотрёт свою анимацию
        if self._spinner_message:
            self._status.console.print(
                f"[dim]⠿[/dim] [blue]{self._spinner_message}[/blue]"
            )
        self._status.stop()
        self._status = None

    # ------------------------------------------------------------------
    # Progress bar
    # ------------------------------------------------------------------

    def show_progress_bar(
        self,
        total: int = 100,
        description: str = "Processing",
        unit: str = "items",
    ) -> int:
        """Показать прогресс-бар и вернуть ID задачи."""
        self.stop_spinner()

        # Свежая консоль чтобы Progress не конфликтовал с предыдущим Status
        self._console = self._fresh_console()
        self._progress = Progress(
            TextColumn("[bold blue]{task.description}"),
            BarColumn(bar_width=None),
            TaskProgressColumn(),
            TimeRemainingColumn(),
            console=self._console,
            expand=True,
        )
        task_id = self._progress.add_task(description, total=total, unit=unit)
        self._progress.start()
        return task_id

    def update_progress_bar(
        self,
        task_id: int,
        advance: int = 1,
        description: Optional[str] = None,
    ) -> None:
        """Обновить прогресс-бар."""
        if self._progress is None:
            return
        self._progress.update(task_id, advance=advance)
        if description:
            self._progress.update(task_id, description=description)

    def complete_progress_bar(self, task_id: int, description: str = "Completed") -> None:
        """Завершить прогресс-бар."""
        if self._progress is None:
            return
        try:
            task = self._progress.tasks[task_id]
            self._progress.update(task_id, completed=task.total, description=description)
            self._progress.stop()
        except (IndexError, KeyError):
            pass
        self._progress = None
        # Сбрасываем консоль — Progress оставляет _live в грязном состоянии
        self._console = self._fresh_console()

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def log(self, message: str, level: str = "info") -> None:
        """Вывести лог-сообщение."""
        ts = datetime.now().strftime("%H:%M:%S")

        styles = {
            "info":    f"[dim][{ts}][/dim] {message}",
            "success": f"[green][✓] {message}[/green]",
            "warning": f"[yellow][!] {message}[/yellow]",
            "error":   f"[red][✗] {message}[/red]",
        }
        self._safe_print(styles.get(level, message))

    # ------------------------------------------------------------------
    # Result output
    # ------------------------------------------------------------------

    def print_result(self, title: str, content: str) -> None:
        """Красиво вывести результат."""
        from rich.panel import Panel

        self.stop_spinner()
        self._console.print(f"\n[bold]{title}[/bold]")
        self._console.print(Panel(content, border_style="blue", padding=(1, 2)))


# Глобальный экземпляр для удобного использования
progress_display = ProgressDisplay()