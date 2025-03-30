"""
Code written in Python to solve Sudoku
The key idea is to implement the function using many functions. The program is structured by solving 3 key problem:
*   Finding Unassigned Entry of the array (the zeros which are equivalent to the blank spaces of unresolved sudokus)
*   Checking if it is safe to put a number (there are not repetition in the coloumn, in the row or in the cell)
*   Checking for every cell every number
a is the sudoku matrix ~ sudoku is its name in the function
"""

a = [[9, 0, 1, 0, 0, 5, 4, 8, 0],
     [0, 0, 0, 2, 0, 0, 0, 7, 0],
     [0, 8, 0, 0, 0, 0, 0, 0, 0],
     [4, 0, 6, 0, 0, 9, 1, 0, 0],
     [3, 0, 0, 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 5, 0, 0, 0, 9],
     [6, 0, 8, 0, 7, 0, 0, 4, 0],
     [0, 5, 0, 0, 0, 0, 8, 0, 0],
     [0, 3, 0, 0, 0, 6, 0, 0, 0]]

def find_unassigned_entry(sudoku):
    """
    find_unassigned_entry: returns the first ij position of the matrix cell which has zero (= blank space) in it. If he doesn't exists it returns 0
    Arguments:
    - sudoku: matrix which represents the sudoku
    Output:
    - x, y: ij position of the cell which has zero in it
    otherwise it returns False if there aren't cell wih zero in it
    """
    for x in range(9):
        for y in range(9):
            if sudoku[x][y] == 0:
                return x, y
    return None, None

def is_safe(sudoku, n, x, y):
    """
    is_safe: return True if a number can be put in a position
    """
    return (check_row(sudoku, n, x) and
            check_col(sudoku, n, y) and
            check_box(sudoku, n, x, y))

def check_row(sudoku, n, x):
    """
    check_row: function that iterates in a row to see if there is a repetion and in that case returns False
    """
    return n not in sudoku[x]

def check_col(sudoku, n, y):
    """
    check_col: function that iterates in a coulumn to see if there is a repetition and in that case return False
    """
    return all(sudoku[x][y] != n for x in range(9))

def check_box(sudoku, n, x, y):
    """
    check_box: function that iterates in a box (by using the approx of the ratio between x and 3 and the same for y to
    find the upper-left corner of the cell we are currently in) to see if there are repetitions
    """
    x_start = (x // 3) * 3
    y_start = (y // 3) * 3
    for delta_x in range(3):
        for delta_y in range(3):
            if sudoku[x_start + delta_x][y_start + delta_y] == n:
                return False
    return True
