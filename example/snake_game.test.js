import assert from "node:assert/strict";

import { DIRECTIONS, advanceGame, createInitialState, placeFood, queueDirection } from "./snake_game.js";

let failures = 0;
let total = 0;

function runTest(name, callback) {
  total += 1;

  try {
    callback();
    console.log(`PASS ${name}`);
  } catch (error) {
    failures += 1;
    console.error(`FAIL ${name}`);
    console.error(error.stack);
  }
}

runTest("snake advances one cell in the active direction", () => {
  const state = createInitialState({
    gridSize: 8,
    snake: [
      { x: 3, y: 3 },
      { x: 2, y: 3 },
      { x: 1, y: 3 },
    ],
    direction: DIRECTIONS.RIGHT,
    food: { x: 0, y: 0 },
  });

  const nextState = advanceGame(state, () => 0);

  assert.deepEqual(nextState.snake, [
    { x: 4, y: 3 },
    { x: 3, y: 3 },
    { x: 2, y: 3 },
  ]);
  assert.equal(nextState.score, 0);
  assert.equal(nextState.gameOver, false);
});

runTest("reverse turns are ignored", () => {
  const state = createInitialState({
    gridSize: 8,
    snake: [
      { x: 3, y: 3 },
      { x: 2, y: 3 },
      { x: 1, y: 3 },
    ],
    direction: DIRECTIONS.RIGHT,
    food: { x: 0, y: 0 },
  });

  const nextState = queueDirection(state, DIRECTIONS.LEFT);

  assert.equal(nextState.nextDirection, DIRECTIONS.RIGHT);
});

runTest("eating food grows the snake and increases the score", () => {
  const state = createInitialState({
    gridSize: 6,
    snake: [
      { x: 2, y: 2 },
      { x: 1, y: 2 },
      { x: 0, y: 2 },
    ],
    direction: DIRECTIONS.RIGHT,
    food: { x: 3, y: 2 },
  });

  const nextState = advanceGame(state, () => 0);

  assert.equal(nextState.snake.length, 4);
  assert.equal(nextState.score, 1);
  assert.deepEqual(nextState.snake[0], { x: 3, y: 2 });
  assert.notDeepEqual(nextState.food, { x: 3, y: 2 });
  assert.equal(
    nextState.snake.some(
      (segment) =>
        nextState.food &&
        segment.x === nextState.food.x &&
        segment.y === nextState.food.y
    ),
    false
  );
});

runTest("moving beyond the board ends the game", () => {
  const state = createInitialState({
    gridSize: 4,
    snake: [
      { x: 3, y: 1 },
      { x: 2, y: 1 },
      { x: 1, y: 1 },
    ],
    direction: DIRECTIONS.RIGHT,
    food: { x: 0, y: 0 },
  });

  const nextState = advanceGame(state, () => 0);

  assert.equal(nextState.gameOver, true);
});

runTest("running into the body ends the game", () => {
  const state = createInitialState({
    gridSize: 6,
    snake: [
      { x: 2, y: 2 },
      { x: 3, y: 2 },
      { x: 3, y: 1 },
      { x: 2, y: 1 },
      { x: 1, y: 1 },
    ],
    direction: DIRECTIONS.UP,
    food: { x: 5, y: 5 },
  });

  const nextState = advanceGame(state, () => 0);

  assert.equal(nextState.gameOver, true);
});

runTest("food placement only uses empty cells", () => {
  const snake = [
    { x: 0, y: 0 },
    { x: 1, y: 0 },
    { x: 2, y: 0 },
    { x: 0, y: 1 },
    { x: 1, y: 1 },
    { x: 2, y: 1 },
    { x: 0, y: 2 },
    { x: 1, y: 2 },
  ];

  const food = placeFood(snake, 3, () => 0.99);

  assert.deepEqual(food, { x: 2, y: 2 });
});

if (failures > 0) {
  console.error(`\n${failures} of ${total} snake logic tests failed.`);
  process.exit(1);
}

console.log(`\n${total} snake logic tests passed.`);
