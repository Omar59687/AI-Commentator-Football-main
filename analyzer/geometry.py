from math import sqrt


def ball_center(ball_bbox):
    x1, y1, x2, y2 = ball_bbox
    return ((x1 + x2) / 2, (y1 + y2) / 2)


def player_feet(player_bbox):
    x1, y1, x2, y2 = player_bbox
    return ((x1 + x2) / 2, y2)


def distance(player_bbox, ball_bbox):

    px, py = player_feet(player_bbox)
    bx, by = ball_center(ball_bbox)

    return sqrt(
        (px - bx) ** 2 +
        (py - by) ** 2
    )