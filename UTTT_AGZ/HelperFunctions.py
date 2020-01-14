import numpy as np


def index_2d_to_1d_array(x, y):
    return (x * 9) + y


def index_1d_to_2d_array(index):
    x = index // 9
    y = index % 9
    return x, y


def reshape_to_sudoku_matrix(matrix):
    return matrix.reshape(3, 3, 3, 3).swapaxes(1, 2).reshape(9, 9)


def update_index_after_reshape(old_index):
    index_matrix = [    [[0,0],[0,1],[0,2],[1,0],[1,1],[1,2],[2,0],[2,1],[2,2]],
                        [[0,3],[0,4],[0,5],[1,3],[1,4],[1,5],[2,3],[2,4],[2,5]],
                        [[0,6],[0,7],[0,8],[1,6],[1,7],[1,8],[2,6],[2,7],[2,8]],
                        [[3,0],[3,1],[3,2],[4,0],[4,1],[4,2],[5,0],[5,1],[5,2]],
                        [[3,3],[3,4],[3,5],[4,3],[4,4],[4,5],[5,3],[5,4],[5,5]],
                        [[3,6],[3,7],[3,8],[4,6],[4,7],[4,8],[5,6],[5,7],[5,8]],
                        [[6,0],[6,1],[6,2],[7,0],[7,1],[7,2],[8,0],[8,1],[8,2]],
                        [[6,3],[6,4],[6,5],[7,3],[7,4],[7,5],[8,3],[8,4],[8,5]],
                        [[6,6],[6,7],[6,8],[7,6],[7,7],[7,8],[8,6],[8,7],[8,8]]
                    ]
    return index_matrix[old_index[0]][old_index[1]]
