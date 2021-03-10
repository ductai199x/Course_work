class Solution(object):
    def inRange(self, grid, r, c):
        numRow, numCol = len(grid), len(grid[0])
        if r < 0 or c < 0 or r >= numRow or c >= numCol:
            return False
        return True

    def numIslands(self, grid):
        num_row, num_col = len(grid), len(grid[0])
        num_island = 0

        for i in range(0, num_row):
            for j in range(0, num_col):
                if grid[i][j] == 1:
                    self.dfs(grid, i, j)
                    num_island += 1

        return num_island

    def dfs(self, grid, i, j):
        # up, right, left, down
        directions = [[0,1], [1,0], [-1,0], [0, -1]]
        
        # mark current cell as visited
        grid[i][j] = 2

        for d in directions:
            next_row, next_col = i + d[0], j + d[1]
            if self.inRange(grid, next_row, next_col) and grid[next_row][next_col] == 1:
                self.dfs(grid, next_row, next_col)



grid = [[1, 1, 0, 1, 0],
        [0, 1, 0, 0, 1],
        [1, 0, 0, 1, 1],
        [0, 0, 0, 0, 0]]
print(Solution().numIslands(grid))
# 3