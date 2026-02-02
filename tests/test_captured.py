import pytest
from go import Go
import time


def test_captured():
    start_time = time.perf_counter()
    go = Go(9)
    # should probably hide the board but oh well
    input_board = """
B - B - O - B - B - B - O - B - B
|   |   |   |   |   |   |   |   |
B - O - O - W - O - O - W - W - B
|   |   |   |   |   |   |   |   |
O - B - W - O - B - B - W - B - O
|   |   |   |   |   |   |   |   |
B - W - W - W - B - B - B - O - W
|   |   |   |   |   |   |   |   |
W - W - B - W - O - B - W - O - O
|   |   |   |   |   |   |   |   |
B - O - O - W - W - W - O - W - W
|   |   |   |   |   |   |   |   |
W - W - B - W - B - W - W - B - W
|   |   |   |   |   |   |   |   |
W - O - O - B - B - W - B - B - O
|   |   |   |   |   |   |   |   |
B - O - W - B - B - W - O - W - O"""

    lines = input_board.strip().split("\n")
    go.board = []
    for line in lines:
        if "|" not in line:  # This is a data line
            row = line.split(" - ")  # Split by " - " to get individual elements
            go.board.append(row)

    # Print the resulting 2D array
    print("\n")
    go.play_move("W", 4, 8)
    expected_string = """
B - B - O - B - B - B - O - B - B
|   |   |   |   |   |   |   |   |
B - O - O - W - O - O - W - W - B
|   |   |   |   |   |   |   |   |
O - B - W - O - B - B - W - B - O
|   |   |   |   |   |   |   |   |
B - W - W - W - B - B - B - O - W
|   |   |   |   |   |   |   |   |
W - W - B - W - O - B - W - O - W
|   |   |   |   |   |   |   |   |
B - O - O - W - W - W - O - W - W
|   |   |   |   |   |   |   |   |
W - W - B - W - B - W - W - B - W
|   |   |   |   |   |   |   |   |
W - O - O - B - B - W - B - B - O
|   |   |   |   |   |   |   |   |
B - O - W - B - B - W - O - W - O"""
    lines = expected_string.strip().split("\n")

    expected = []
    for line in lines:
        if "|" not in line:  # This is a data line
            row = line.split(" - ")  # Split by " - " to get individual elements
            expected.append(row)
    expected_go = Go(9)
    expected_go.board = expected
    expected_go.check_all_captured("B", "W")
    expected_go.print_board()

    end_time = time.perf_counter()
    elapsed_time = end_time - start_time

    print(f"Elapsed time: {elapsed_time:.4f} seconds")
    assert go.board == expected


def test_captured2():
    go = Go(9)
    input_board = """
B - B - O - B - B - B - O - B - B
|   |   |   |   |   |   |   |   |
B - O - O - W - O - O - W - W - B
|   |   |   |   |   |   |   |   |
O - B - W - O - B - B - W - B - O
|   |   |   |   |   |   |   |   |
B - W - W - W - B - B - B - O - W
|   |   |   |   |   |   |   |   |
W - W - B - W - O - B - W - O - W
|   |   |   |   |   |   |   |   |
B - O - O - W - W - W - O - W - W
|   |   |   |   |   |   |   |   |
W - W - B - W - B - W - W - B - W
|   |   |   |   |   |   |   |   |
W - O - B - B - B - W - B - B - O
|   |   |   |   |   |   |   |   |
B - O - W - B - O - W - O - W - O
"""
    lines = input_board.strip().split("\n")
    go.board = []
    for line in lines:
        if "|" not in line:  # This is a data line
            row = line.split(" - ")  # Split by " - " to get individual elements
            go.board.append(row)

    # Print the resulting 2D array
    print("\n")
    go.play_move("W", 1, 5)
    captured = go.check_all_captured("B", "W")
    assert captured == []


def test_captured3():
    go = Go(9)
    # should probably hide the board but oh well
    input_board = """
O - O - W - O - O - O - O - O - O
|   |   |   |   |   |   |   |   |
O - O - B - O - O - O - O - O - O
|   |   |   |   |   |   |   |   |
O - O - B - O - O - O - B - O - O
|   |   |   |   |   |   |   |   |
O - O - W - O - O - O - B - B - O
|   |   |   |   |   |   |   |   |
O - W - B - B - B - O - W - O - W
|   |   |   |   |   |   |   |   |
O - W - B - W - O - O - O - O - O
|   |   |   |   |   |   |   |   |
O - W - W - B - O - B - O - O - O
|   |   |   |   |   |   |   |   |
O - O - O - O - O - O - O - O - O
|   |   |   |   |   |   |   |   |
O - O - O - W - O - O - O - O - O"""

    lines = input_board.strip().split("\n")
    go.board = []
    for line in lines:
        if "|" not in line:  # This is a data line
            row = line.split(" - ")  # Split by " - " to get individual elements
            go.board.append(row)

    # Print the resulting 2D array
    print("\n")
    go.play_move("W", 1, 4)
    expected_string = """
O - O - W - O - O - O - O - O - O
|   |   |   |   |   |   |   |   |
O - O - B - O - W - O - O - O - O
|   |   |   |   |   |   |   |   |
O - O - B - O - O - O - B - O - O
|   |   |   |   |   |   |   |   |
O - O - W - O - O - O - B - B - O
|   |   |   |   |   |   |   |   |
O - W - B - B - B - O - W - O - W
|   |   |   |   |   |   |   |   |
O - W - B - W - O - O - O - O - O
|   |   |   |   |   |   |   |   |
O - W - W - B - O - B - O - O - O
|   |   |   |   |   |   |   |   |
O - O - O - O - O - O - O - O - O
|   |   |   |   |   |   |   |   |
O - O - O - W - O - O - O - O - O"""
    lines = expected_string.strip().split("\n")

    expected = []
    for line in lines:
        if "|" not in line:  # This is a data line
            row = line.split(" - ")  # Split by " - " to get individual elements
            expected.append(row)
    expected_go = Go(9)
    expected_go.board = expected
    expected_go.check_all_captured("B", "W")
    expected_go.print_board()
    assert go.board == expected
