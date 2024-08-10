#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <SDL2/SDL.h>
#include <string.h>

#define W 30
#define H 20
#define S_SIZE 256
#define N_ACTS 4

#define CELL_SIZE 30
#define WINDOW_WIDTH (W * CELL_SIZE)
#define WINDOW_HEIGHT (H * CELL_SIZE)

#define DEFAULT_DELAY 50
#define DEFAULT_E_START 1.0
#define DEFAULT_E_END 0.0001
#define DEFAULT_N_EPS 200000
#define DEFAULT_LEARNING_RATE 0.1
#define DEFAULT_DISCOUNT_FACTOR 0.99
#define DEFAULT_EPSILON_DECAY 0.9999

#define DEFAULT_FOOD_REWARD 1.0
#define DEFAULT_WALL_PENALTY -1.0
#define DEFAULT_STEP_PENALTY -0.025

typedef struct {
    int x, y;
} Pt;

typedef struct {
    Pt *body;
    int len;
    int dir;
    int score;
    int steps;
} Snk;

typedef struct {
    double learning_rate;
    double discount_factor;
    double epsilon_decay;
    int delay;
    double e_start;
    double e_end;
    int n_eps;
    double food_reward;
    double wall_penalty;
    double step_penalty;
} Params;

Snk snk;
Pt food;
int game_over;

double ***qtable;

SDL_Window *window = NULL;
SDL_Renderer *renderer = NULL;

void setup() {
    snk.len = 1;
    snk.body = malloc(sizeof(Pt) * (W * H));
    snk.body[0] = (Pt){W / 2, H / 2};
    snk.dir = 0;
    snk.score = 0;
    snk.steps = 0;
    food = (Pt){rand() % W, rand() % H};
    game_over = 0;
}

void cleanup() {
    free(snk.body);
}

void alloc_qtable() {
    qtable = malloc(W * H * sizeof(double**));
    for (int i = 0; i < W * H; i++) {
        qtable[i] = malloc(S_SIZE * sizeof(double*));
        for (int j = 0; j < S_SIZE; j++) {
            qtable[i][j] = calloc(N_ACTS, sizeof(double));
        }
    }
}

void free_qtable() {
    for (int i = 0; i < W * H; i++) {
        for (int j = 0; j < S_SIZE; j++) {
            free(qtable[i][j]);
        }
        free(qtable[i]);
    }
    free(qtable);
}

void move_snake() {
    for (int i = snk.len - 1; i > 0; i--) {
        snk.body[i] = snk.body[i-1];
    }
    static const int dx[] = {0, 1, 0, -1};
    static const int dy[] = {-1, 0, 1, 0};
    snk.body[0].x += dx[snk.dir];
    snk.body[0].y += dy[snk.dir];
    snk.steps++;
}

int check_collision() {
    Pt head = snk.body[0];
    if (head.x < 0 || head.x >= W || head.y < 0 || head.y >= H) {
        return 1;
    }
    for (int i = 1; i < snk.len; i++) {
        if (head.x == snk.body[i].x && head.y == snk.body[i].y) {
            return 1;
        }
    }
    return 0;
}

int check_food() {
    if (snk.body[0].x == food.x && snk.body[0].y == food.y) {
        snk.len++;
        snk.score++;
        int placed = 0;
        while (!placed) {
            food = (Pt){rand() % W, rand() % H};
            placed = 1;
            for (int i = 0; i < snk.len; i++) {
                if (food.x == snk.body[i].x && food.y == snk.body[i].y) {
                    placed = 0;
                    break;
                }
            }
        }
        return 1;
    }
    return 0;
}

void render() {
    SDL_SetRenderDrawColor(renderer, 0, 0, 0, 255);
    SDL_RenderClear(renderer);

    SDL_SetRenderDrawColor(renderer, 50, 50, 50, 255);
    for (int i = 0; i <= W; i++) {
        SDL_RenderDrawLine(renderer, i * CELL_SIZE, 0, i * CELL_SIZE, WINDOW_HEIGHT);
    }
    for (int i = 0; i <= H; i++) {
        SDL_RenderDrawLine(renderer, 0, i * CELL_SIZE, WINDOW_WIDTH, i * CELL_SIZE);
    }

    SDL_SetRenderDrawColor(renderer, 255, 0, 0, 255);
    SDL_Rect food_rect = {food.x * CELL_SIZE, food.y * CELL_SIZE, CELL_SIZE, CELL_SIZE};
    SDL_RenderFillRect(renderer, &food_rect);

    for (int i = 0; i < snk.len; i++) {
        SDL_SetRenderDrawColor(renderer, i == 0 ? 0 : 0, 255, i == 0 ? 0 : 200, 255);
        SDL_Rect snake_rect = {snk.body[i].x * CELL_SIZE, snk.body[i].y * CELL_SIZE, CELL_SIZE, CELL_SIZE};
        SDL_RenderFillRect(renderer, &snake_rect);
    }

    SDL_RenderPresent(renderer);
}

int get_state() {
    int dx = food.x - snk.body[0].x;
    int dy = food.y - snk.body[0].y;
    int state = 0;
    state |= (dx > 0) ? 1 : (dx < 0) ? 2 : 0;
    state |= (dy > 0) ? 4 : (dy < 0) ? 8 : 0;
    for (int i = 0; i < 4; i++) {
        int nx = snk.body[0].x + (i == 1) - (i == 3);
        int ny = snk.body[0].y + (i == 2) - (i == 0);
        if (nx < 0 || nx >= W || ny < 0 || ny >= H) {
            state |= (1 << (i + 4));
        } else {
            for (int j = 1; j < snk.len; j++) {
                if (nx == snk.body[j].x && ny == snk.body[j].y) {
                    state |= (1 << (i + 4));
                    break;
                }
            }
        }
    }
    return state;
}

int choose_action(double e) {
    if ((double)rand() / RAND_MAX < e) {
        return rand() % N_ACTS;
    } else {
        int state = get_state();
        int pos = snk.body[0].y * W + snk.body[0].x;
        int best = 0;
        double best_val = qtable[pos][state][0];
        for (int i = 1; i < N_ACTS; i++) {
            if (qtable[pos][state][i] > best_val) {
                best_val = qtable[pos][state][i];
                best = i;
            }
        }
        return best;
    }
}

void update_qtable(int ox, int oy, int os, int a, double r, int nx, int ny, Params* params) {
    int ns = get_state();
    int old_pos = oy * W + ox;
    int new_pos = ny * W + nx;

    old_pos = (old_pos < 0) ? 0 : (old_pos >= W * H) ? W * H - 1 : old_pos;
    new_pos = (new_pos < 0) ? 0 : (new_pos >= W * H) ? W * H - 1 : new_pos;

    os = (os < 0) ? 0 : (os >= S_SIZE) ? S_SIZE - 1 : os;
    ns = (ns < 0) ? 0 : (ns >= S_SIZE) ? S_SIZE - 1 : ns;

    a = (a < 0) ? 0 : (a >= N_ACTS) ? N_ACTS - 1 : a;

    double max_q = qtable[new_pos][ns][0];
    for (int i = 1; i < N_ACTS; i++) {
        if (qtable[new_pos][ns][i] > max_q) {
            max_q = qtable[new_pos][ns][i];
        }
    }

    qtable[old_pos][os][a] += params->learning_rate * (r + params->discount_factor * max_q - qtable[old_pos][os][a]);
}

void save_qtable(const char* filename) {
    FILE* file = fopen(filename, "wb");
    if (file == NULL) {
        fprintf(stderr, "Error opening file for writing\n");
        return;
    }

    for (int i = 0; i < W * H; i++) {
        for (int j = 0; j < S_SIZE; j++) {
            fwrite(qtable[i][j], sizeof(double), N_ACTS, file);
        }
    }

    fclose(file);
    printf("Q-table saved to %s\n", filename);
}

void load_qtable(const char* filename) {
    FILE* file = fopen(filename, "rb");
    if (file == NULL) {
        fprintf(stderr, "Error opening file for reading\n");
        return;
    }

    for (int i = 0; i < W * H; i++) {
        for (int j = 0; j < S_SIZE; j++) {
            fread(qtable[i][j], sizeof(double), N_ACTS, file);
        }
    }

    fclose(file);
    printf("Q-table loaded from %s\n", filename);
}

void learn(Params* params) {
    double e = params->e_start;
    int max_score = 0;
    int total_score = 0;
    int episodes_since_improvement = 0;

    for (int ep = 0; ep < params->n_eps; ep++) {
        setup();

        while (!game_over) {
            int ox = snk.body[0].x;
            int oy = snk.body[0].y;
            int os = get_state();

            int a = choose_action(e);
            snk.dir = a;

            move_snake();
            game_over = check_collision();
            int ate = check_food();

            int nx = snk.body[0].x;
            int ny = snk.body[0].y;

            double r = game_over ? params->wall_penalty : (ate ? params->food_reward : params->step_penalty);

            update_qtable(ox, oy, os, a, r, nx, ny, params);

            if (snk.steps > W * H * 2) {
                game_over = 1;
            }
        }

        total_score += snk.score;
        if (snk.score > max_score) {
            max_score = snk.score;
            episodes_since_improvement = 0;
        } else {
            episodes_since_improvement++;
        }

        if (ep % 1000 == 0) {
            double avg_score = (double)total_score / 1000;
            printf("Ep %d, Avg Score: %.2f, Max: %d, E: %.4f\n", ep, avg_score, max_score, e);
            total_score = 0;
        }

        if (episodes_since_improvement > 10000) {
            e = fmin(params->e_start, e / params->epsilon_decay);
            episodes_since_improvement = 0;
        } else {
            e = fmax(params->e_end, e * params->epsilon_decay);
        }

        cleanup();
    }

    save_qtable("snake_qtable.bin");
}

int main(int argc, char* argv[]) {
    srand(time(NULL));

    int load_file = 0;
    char* filename = NULL;
    Params params = {
        DEFAULT_LEARNING_RATE,
        DEFAULT_DISCOUNT_FACTOR,
        DEFAULT_EPSILON_DECAY,
        DEFAULT_DELAY,
        DEFAULT_E_START,
        DEFAULT_E_END,
        DEFAULT_N_EPS,
        DEFAULT_FOOD_REWARD,
        DEFAULT_WALL_PENALTY,
        DEFAULT_STEP_PENALTY
    };

    for (int i = 1; i < argc; i++) {
        if (strncmp(argv[i], "load_file=", 10) == 0) {
            load_file = (strcmp(argv[i] + 10, "True") == 0);
        } else if (strncmp(argv[i], "file=", 5) == 0) {
            filename = argv[i] + 5;
        } else if (strncmp(argv[i], "learning_rate=", 14) == 0) {
            params.learning_rate = atof(argv[i] + 14);
        } else if (strncmp(argv[i], "discount_factor=", 16) == 0) {
            params.discount_factor = atof(argv[i] + 16);
        } else if (strncmp(argv[i], "epsilon_decay=", 14) == 0) {
            params.epsilon_decay = atof(argv[i] + 14);
        } else if (strncmp(argv[i], "delay=", 6) == 0) {
            params.delay = atoi(argv[i] + 6);
        } else if (strncmp(argv[i], "e_start=", 8) == 0) {
            params.e_start = atof(argv[i] + 8);
        } else if (strncmp(argv[i], "e_end=", 6) == 0) {
            params.e_end = atof(argv[i] + 6);
        } else if (strncmp(argv[i], "n_eps=", 6) == 0) {
            params.n_eps = atoi(argv[i] + 6);
        } else if (strncmp(argv[i], "food_reward=", 12) == 0) {
            params.food_reward = atof(argv[i] + 12);
        } else if (strncmp(argv[i], "wall_penalty=", 13) == 0) {
            params.wall_penalty = atof(argv[i] + 13);
        } else if (strncmp(argv[i], "step_penalty=", 13) == 0) {
            params.step_penalty = atof(argv[i] + 13);
        }
    }

    alloc_qtable();

    if (load_file && filename) {
        load_qtable(filename);
    } else {
        learn(&params);
    }

    if (SDL_Init(SDL_INIT_VIDEO) < 0) {
        fprintf(stderr, "SDL could not initialize! SDL_Error: %s\n", SDL_GetError());
        free_qtable();
        return 1;
    }

    window = SDL_CreateWindow("Snake AI", SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED, WINDOW_WIDTH, WINDOW_HEIGHT, SDL_WINDOW_SHOWN);
    if (!window) {
        fprintf(stderr, "Window could not be created! SDL_Error: %s\n", SDL_GetError());
        SDL_Quit();
        free_qtable();
        return 1;
    }

    renderer = SDL_CreateRenderer(window, -1, SDL_RENDERER_ACCELERATED);
    if (!renderer) {
        fprintf(stderr, "Renderer could not be created! SDL_Error: %s\n", SDL_GetError());
        SDL_DestroyWindow(window);
        SDL_Quit();
        free_qtable();
        return 1;
    }

    setup();
    SDL_Event e;
    int quit = 0;
    while (!quit && !game_over) {
        while (SDL_PollEvent(&e) != 0) {
            if (e.type == SDL_QUIT) {
                quit = 1;
            }
        }

        int a = choose_action(0);

        snk.dir = a;
        move_snake();
        game_over = check_collision();
        check_food();
        render();

        SDL_Delay(DELAY);
    }

    SDL_DestroyRenderer(renderer);
    SDL_DestroyWindow(window);
    SDL_Quit();

    cleanup();
    free_qtable();

    printf("Game over. Final score: %d\n", snk.score);

    return 0;
}
