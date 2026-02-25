def simple_planner(grid, start, goal):
    path = []
    x, y = start
    gx, gy = goal

    while (x, y) != (gx, gy):
        if x < gx:
            x += 1
        elif x > gx:
            x -= 1
        elif y < gy:
            y += 1
        elif y > gy:
            y -= 1

        path.append((x, y))

    return path 