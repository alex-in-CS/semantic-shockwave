import assert from 'node:assert/strict';
import { test } from 'node:test';

import {
  challengeFor, chainCost, dayNumber, loadHistory, saveResult, scoreChain, shareText, streak,
} from '../src/game.js';

// Concepts on a unit circle: angle in degrees -> unit vector.
const at = (deg) => [Math.cos((deg * Math.PI) / 180), Math.sin((deg * Math.PI) / 180)];
const VECTORS = { a: at(0), b: at(30), c: at(60), d: at(90), x: at(40), far: at(200) };
const vectorOf = (id) => VECTORS[id];

test('day numbers count from the epoch', () => {
  assert.equal(dayNumber('2026-10-01'), 1);
  assert.equal(dayNumber('2026-10-31'), 31);
  assert.equal(dayNumber('2027-10-01'), 366);
});

test('challengeFor uses the schedule, then a stable fallback', () => {
  const schedule = [{ date: '2026-10-01', source: 'a', target: 'd' }, { date: '2026-10-02', source: 'b', target: 'c' }];
  assert.equal(challengeFor(schedule, '2026-10-02').source, 'b');
  const later = challengeFor(schedule, '2030-01-01');
  assert.equal(later.date, '2030-01-01');
  assert.deepEqual(challengeFor(schedule, '2030-01-01'), later);
  assert.equal(challengeFor([], '2026-10-01'), null);
});

test('matching the optimal chain scores 100 with all green', () => {
  const r = scoreChain(['a', 'b', 'c', 'd'], ['a', 'b', 'c', 'd'], vectorOf);
  assert.equal(r.score, 100);
  assert.equal(r.beatOptimal, false);
  assert.deepEqual(r.marks, ['🟩', '🟩']);
});

test('worse chains score lower; near misses are yellow', () => {
  const optimal = ['a', 'b', 'c', 'd'];
  const near = scoreChain(optimal, ['a', 'x', 'c', 'd'], vectorOf);
  const bad = scoreChain(optimal, ['a', 'far', 'd'], vectorOf);
  assert.ok(near.score < 100 && near.score > bad.score, `${near.score} vs ${bad.score}`);
  assert.deepEqual(near.marks, ['🟨', '🟩']);
  assert.deepEqual(bad.marks, ['⬛']);
});

test('a cheaper chain beats the algorithm', () => {
  // The optimal path took one big leap; the player found the small steps.
  const r = scoreChain(['a', 'd'], ['a', 'b', 'c', 'd'], vectorOf);
  assert.equal(r.beatOptimal, true);
  assert.equal(r.score, 100);
  assert.ok(chainCost(['a', 'b', 'c', 'd'], vectorOf) < chainCost(['a', 'd'], vectorOf));
});

test('share text carries the number, pair, marks and score', () => {
  const result = { score: 87, beatOptimal: false, marks: ['🟩', '⬛'] };
  const text = shareText({ number: 3, source: 'coffee', target: 'revolution', result, url: 'https://x' });
  assert.match(text, /#3/);
  assert.match(text, /coffee → revolution/);
  assert.match(text, /🟩⬛ 87%/);
});

test('history survives broken storage and counts streaks', () => {
  const memory = new Map();
  const storage = { getItem: (k) => memory.get(k) ?? null, setItem: (k, v) => memory.set(k, v) };
  saveResult('2026-10-01', { score: 50 }, storage);
  saveResult('2026-10-02', { score: 70 }, storage);
  const history = loadHistory(storage);
  assert.equal(streak(history, '2026-10-02'), 2);
  assert.equal(streak(history, '2026-10-03'), 2, "today not played yet: yesterday's streak holds");
  assert.equal(streak(history, '2026-10-05'), 0);

  const broken = { getItem: () => { throw new Error('denied'); }, setItem: () => { throw new Error('denied'); } };
  assert.deepEqual(loadHistory(broken), {});
  assert.doesNotThrow(() => saveResult('2026-10-01', { score: 1 }, broken));
});
