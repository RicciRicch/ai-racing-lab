import pygame


# ==================================================
# COLORS
# ==================================================

GRASS_COLOR = (
    40,
    130,
    60
)

ROAD_COLOR = (
    70,
    70,
    70
)


# ==================================================
# TRACK CLASS
# ==================================================

class Track:
    def __init__(
        self,
        name,
        start_x,
        start_y,
        start_angle,
        checkpoints,
        draw_function
    ):
        self.name = name

        self.start_x = start_x
        self.start_y = start_y
        self.start_angle = start_angle

        self.checkpoints = checkpoints

        self.draw_function = (
            draw_function
        )

    def draw(self, screen):
        screen.fill(
            GRASS_COLOR
        )

        self.draw_function(
            screen
        )


# ==================================================
# TRACK 1
# Original racing oval
# ==================================================

def draw_track_1(screen):
    pygame.draw.ellipse(
        screen,
        ROAD_COLOR,
        (
            100,
            70,
            800,
            560
        )
    )

    pygame.draw.ellipse(
        screen,
        GRASS_COLOR,
        (
            280,
            200,
            440,
            300
        )
    )


track_1 = Track(
    name="Classic Oval",

    start_x=500,
    start_y=590,
    start_angle=-90,

    checkpoints=[
        # Right
        pygame.Rect(
            760,
            320,
            20,
            60
        ),

        # Top
        pygame.Rect(
            470,
            150,
            60,
            20
        ),

        # Left
        pygame.Rect(
            220,
            320,
            20,
            60
        ),

        # Bottom
        pygame.Rect(
            470,
            520,
            60,
            20
        ),
    ],

    draw_function=draw_track_1
)


# ==================================================
# TRACK 2
# Taller and narrower oval
# ==================================================

def draw_track_2(screen):
    pygame.draw.ellipse(
        screen,
        ROAD_COLOR,
        (
            160,
            50,
            680,
            600
        )
    )

    pygame.draw.ellipse(
        screen,
        GRASS_COLOR,
        (
            330,
            180,
            340,
            340
        )
    )


track_2 = Track(
    name="Tall Oval",

    start_x=500,
    start_y=615,
    start_angle=-90,

    checkpoints=[
        # Right
        pygame.Rect(
            770,
            320,
            20,
            60
        ),

        # Top
        pygame.Rect(
            470,
            100,
            60,
            20
        ),

        # Left
        pygame.Rect(
            210,
            320,
            20,
            60
        ),

        # Bottom
        pygame.Rect(
            470,
            560,
            60,
            20
        ),
    ],

    draw_function=draw_track_2
)


# ==================================================
# TRACK 3
# Asymmetric oval
# ==================================================

def draw_track_3(screen):
    # Outer road
    pygame.draw.ellipse(
        screen,
        ROAD_COLOR,
        (
            80,
            100,
            840,
            500
        )
    )

    # Inner grass is intentionally shifted
    # to make road width asymmetric.
    pygame.draw.ellipse(
        screen,
        GRASS_COLOR,
        (
            350,
            190,
            390,
            280
        )
    )


track_3 = Track(
    name="Asymmetric Oval",

    start_x=500,
    start_y=570,
    start_angle=-90,

    checkpoints=[
        # Right
        pygame.Rect(
            820,
            320,
            20,
            60
        ),

        # Top
        pygame.Rect(
            470,
            135,
            60,
            20
        ),

        # Left
        pygame.Rect(
            140,
            320,
            20,
            60
        ),

        # Bottom
        pygame.Rect(
            470,
            520,
            60,
            20
        ),
    ],

    draw_function=draw_track_3
)


# ==================================================
# TRAINING TRACKS
# ==================================================

TRAINING_TRACKS = [
    track_1,
    track_2,
    track_3
]