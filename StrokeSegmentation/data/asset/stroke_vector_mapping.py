# this is the map for each character to transform stroke to vector
# the first index in each line shows how many strokes this character has
# then other index means:
# 0 -> exception
# 1 -> vec1
# 2 -> vec2
# 3 -> vec3
# 4 -> vec4
# 5 -> other
# to get the meaning of the vector,
# see direct_vector_def.png and stroke_to_vector_1.png
STROKE_VECTOR_MAP = [
    [5, 2, 3, 1, 2, 4],  # this is '禾'
    [6, 4, 4, 2, 3, 3, 1],  # this is '汗'
    [4, 2, 4, 4, 4],  # this is '心'
    [4, 4, 2, 2, 4],  # this is '火'
    [11, 4, 2, 3, 1, 2, 4, 4, 3, 4, 2, 3],  # this is '粒'
    [11, 3, 1, 2, 4, 4, 3, 4, 2, 3, 3, 1],  # this is '梓'
    [8, 3, 4, 2, 3, 1, 1, 5, 3],  # this is '苦'
    [5, 4, 4, 3, 2, 4],  # this is '头'
    [4, 2, 3, 3, 1],  # this is '手'
    [6, 3, 4, 2, 4, 1, 1],  # this is '划'
    [4, 1, 5, 2, 4],  # this is '水'
    [7, 3, 1, 2, 4, 5, 1, 3],  # this is '李'
    [4, 3, 3, 5, 4],  # this is '云'
    [7, 4, 2, 2, 4, 4, 5, 4],  # this is '冷'
    [4, 2, 5, 2, 4],  # this is '风'
    [5, 2, 5, 4, 1, 4],  # this is '外'
    [4, 3, 1, 2, 5],  # this is '比'
    [8, 2, 1, 5, 3, 3, 2, 5, 4],  # this is '的'
    [8, 3, 1, 5, 3, 5, 3, 3, 1],  # this is '事'
    [16, 1, 3, 2, 5, 4, 5, 4, 2, 4, 4, 5, 3, 3, 1, 2, 4]  # this is '餐'
]