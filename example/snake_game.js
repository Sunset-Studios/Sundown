export const GRID_SIZE = 16;
export const DEFAULT_SNAKE_SPEED = 5;

export const DIRECTIONS = Object.freeze({
  UP: "up",
  DOWN: "down",
  LEFT: "left",
  RIGHT: "right",
});

const DIRECTION_VECTORS = Object.freeze({
  [DIRECTIONS.UP]: { x: 0, y: -1 },
  [DIRECTIONS.DOWN]: { x: 0, y: 1 },
  [DIRECTIONS.LEFT]: { x: -1, y: 0 },
  [DIRECTIONS.RIGHT]: { x: 1, y: 0 },
});

const OPPOSITE_DIRECTIONS = Object.freeze({
  [DIRECTIONS.UP]: DIRECTIONS.DOWN,
  [DIRECTIONS.DOWN]: DIRECTIONS.UP,
  [DIRECTIONS.LEFT]: DIRECTIONS.RIGHT,
  [DIRECTIONS.RIGHT]: DIRECTIONS.LEFT,
});

function copyPoint(point) {
  return { x: point.x, y: point.y };
}

function normalizeSpeed(speed) {
  return Number.isFinite(speed) && speed > 0 ? speed : DEFAULT_SNAKE_SPEED;
}

export function arePointsEqual(a, b) {
  return a.x === b.x && a.y === b.y;
}

export function isInsideGrid(point, gridSize) {
  return point.x >= 0 && point.x < gridSize && point.y >= 0 && point.y < gridSize;
}

export function createInitialSnake(gridSize = GRID_SIZE) {
  const center = Math.floor(gridSize / 2);

  return [
    { x: center, y: center },
    { x: center - 1, y: center },
    { x: center - 2, y: center },
  ];
}

export function listEmptyCells(snake, gridSize) {
  const occupied = new Set(snake.map((segment) => `${segment.x}:${segment.y}`));
  const cells = [];

  for (let y = 0; y < gridSize; y += 1) {
    for (let x = 0; x < gridSize; x += 1) {
      const key = `${x}:${y}`;
      if (!occupied.has(key)) {
        cells.push({ x, y });
      }
    }
  }

  return cells;
}

export function placeFood(snake, gridSize, rng = Math.random) {
  const emptyCells = listEmptyCells(snake, gridSize);

  if (emptyCells.length === 0) {
    return null;
  }

  const randomIndex = Math.min(
    emptyCells.length - 1,
    Math.floor(rng() * emptyCells.length)
  );

  return copyPoint(emptyCells[randomIndex]);
}

export function createInitialState(options = {}) {
  const gridSize = options.gridSize ?? GRID_SIZE;
  const direction = options.direction ?? DIRECTIONS.RIGHT;
  const snake = options.snake?.map(copyPoint) ?? createInitialSnake(gridSize);
  const food = options.food ? copyPoint(options.food) : placeFood(snake, gridSize, options.rng);
  const speed = normalizeSpeed(options.speed);

  return {
    gridSize,
    snake,
    direction,
    nextDirection: direction,
    food,
    speed,
    moveAccumulator: options.moveAccumulator ?? 0,
    score: options.score ?? 0,
    paused: false,
    gameOver: food === null,
    won: food === null,
  };
}

export function queueDirection(state, direction) {
  if (!DIRECTION_VECTORS[direction]) {
    return state;
  }

  if (
    direction === state.direction ||
    OPPOSITE_DIRECTIONS[state.direction] === direction
  ) {
    return state;
  }

  return {
    ...state,
    nextDirection: direction,
  };
}

export function togglePause(state) {
  if (state.gameOver) {
    return state;
  }

  return {
    ...state,
    paused: !state.paused,
  };
}

function advanceSnakeStep(state, rng = Math.random) {
  const direction = state.nextDirection ?? state.direction;
  const movement = DIRECTION_VECTORS[direction];
  const head = state.snake[0];
  const nextHead = {
    x: head.x + movement.x,
    y: head.y + movement.y,
  };

  const isEating = state.food ? arePointsEqual(nextHead, state.food) : false;
  const collisionBody = isEating ? state.snake : state.snake.slice(0, -1);

  const hitWall = !isInsideGrid(nextHead, state.gridSize);
  const hitSelf = collisionBody.some((segment) => arePointsEqual(segment, nextHead));

  if (hitWall || hitSelf) {
    return {
      ...state,
      direction,
      nextDirection: direction,
      gameOver: true,
    };
  }

  const snake = [nextHead, ...collisionBody.map(copyPoint)];
  const food = isEating ? placeFood(snake, state.gridSize, rng) : state.food;
  const won = isEating && food === null;

  return {
    ...state,
    snake,
    direction,
    nextDirection: direction,
    food,
    score: state.score + (isEating ? 1 : 0),
    gameOver: won,
    won,
  };
}

export function advanceGame(state, delta_time = 0, rng = Math.random) {
  if (state.gameOver || state.paused || state.won) {
    return state;
  }

  const speed = normalizeSpeed(state.speed);
  const deltaTime = Number.isFinite(delta_time) ? Math.max(0, delta_time) : 0;
  const stepDuration = 1 / speed;
  let nextState = {
    ...state,
    speed,
    moveAccumulator: (state.moveAccumulator ?? 0) + deltaTime,
  };

  while (nextState.moveAccumulator + Number.EPSILON >= stepDuration) {
    nextState = advanceSnakeStep(
      {
        ...nextState,
        moveAccumulator: nextState.moveAccumulator - stepDuration,
      },
      rng
    );

    if (nextState.gameOver || nextState.won) {
      return {
        ...nextState,
        moveAccumulator: 0,
      };
    }
  }

  return {
    ...nextState,
    moveAccumulator: Math.max(0, nextState.moveAccumulator),
  };
}
