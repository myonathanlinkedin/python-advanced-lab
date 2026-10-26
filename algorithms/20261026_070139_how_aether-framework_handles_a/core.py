from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Set, Any

PAGE_SIZE: int = 4096  # bytes per page, simulation

# Global page registry and ID counter
_page_registry: Dict[int, "Page"] = {}
_page_id_counter: int = 0

def _next_page_id() -> int:
    global _page_id_counter
    _page_id_counter += 1
    return _page_id_counter

@dataclass
class Page:
    data: Any
    refcount: int = 1

class ProcessMemory:
    """
    Simulates a process memory space with copy-on-write semantics.
    """
    def __init__(self) -> None:
        self._pages: Set[int] = set()

    def _ensure_page(self, page_id: int) -> None:
        if page_id not in _page_registry:
            raise KeyError(f"Page {page_id} not allocated")

    def read(self, page_id: int) -> Any:
        """
        Return the data stored in the specified page.
        """
        self._ensure_page(page_id)
        if page_id not in self._pages:
            raise KeyError(f"Page {page_id} not mapped in this process")
        return _page_registry[page_id].data

    def write(self, page_id: int, data: Any) -> None:
        """
        Write data to the specified page, performing copy-on-write if necessary.
        """
        if page_id not in self._pages:
            # Allocate new page
            page = Page(data=data, refcount=1)
            _page_registry[page_id] = page
            self._pages.add(page_id)
            return
        page = _page_registry[page_id]
        if page.refcount > 1:
            # Clone page for this process
            page.refcount -= 1
            new_page_id = _next_page_id()
            new_page = Page(data=data, refcount=1)
            _page_registry[new_page_id] = new_page
            self._pages.remove(page_id)
            self._pages.add(new_page_id)
        else:
            # Exclusive ownership, overwrite
            page.data = data

    def allocate_page(self, data: Any) -> int:
        """
        Allocate a new page with the given data and map it into this process.
        Returns the page ID.
        """
        page_id = _next_page_id()
        _page_registry[page_id] = Page(data=data, refcount=1)
        self._pages.add(page_id)
        return page_id

    def fork(self) -> "ProcessMemory":
        """
        Create a child process that shares all pages with this process.
        """
        child = ProcessMemory()
        for pid in self._pages:
            _page_registry[pid].refcount += 1
            child._pages.add(pid)
        return child

    def memory_usage(self) -> int:
        """
        Return the total memory usage in bytes for this process.
        """
        return len(self._pages) * PAGE_SIZE

    def shared_pages(self) -> int:
        """
        Return the number of pages shared with other processes.
        """
        return sum(1 for pid in self._pages if _page_registry[pid].refcount > 1)

    def total_pages(self) -> int:
        """
        Return the total number of pages allocated in the system.
        """
        return len(_page_registry)

    def page_refcount(self, page_id: int) -> int:
        """
        Return the reference count of a specific page.
        """
        self._ensure_page(page_id)
        return _page_registry[page_id].refcount

    def __repr__(self) -> str:
        return f"<ProcessMemory pages={len(self._pages)}>"
