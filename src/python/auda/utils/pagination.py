from dataclasses import dataclass


@dataclass
class Pagination:
    """Represents pagination information.

    Attributes:
        page: The current page number (1-indexed).
        page_size: The number of items per page.
    """

    page: int
    page_size: int


def get_num_page(total_items: int, page_size: int) -> int:
    """Returns the number of pages needed for pagination.

    Args:
        total_items: Total number of items to paginate.
        page_size: Number of items per page.

    Returns:
        int: Total number of pages required.
    """

    if page_size <= 0:
        raise ValueError('Page size must be greater than zero.')

    return (total_items + page_size - 1) // page_size


def get_offset(page_size: int, page: int) -> int:
    """Returns the offset for pagination based on page size and page number.

    Args:
        page_size: Number of items per page.
        page: Current page number (1-indexed).

    Returns:
        int: Offset for the SQL query to fetch items for the current page.

    Raises:
        ValueError: If page_size is less than or equal to zero or if page is
        less than 1.
    """

    if page_size <= 0:
        raise ValueError('Page size must be greater than zero.')
    if page < 1:
        raise ValueError('Page number must be at least 1.')

    return (page - 1) * page_size
