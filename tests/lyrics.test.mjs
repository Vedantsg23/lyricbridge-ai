// Run with: node --test tests/
import test from 'node:test';
import assert from 'node:assert/strict';
import {
  planWindows, chunksToRows, alignTranslations, fixTimings, buildSrt, buildVtt,
  fmtStamp, fmtClock, isNoise, activeRow, rowsToText, SAMPLE_RATE
} from '../lyrics.js';

function tone(seconds, amp = 0.3) {
  const a = new Float32Array(Math.round(seconds * SAMPLE_RATE));
  for (let i = 0; i < a.length; i++) a[i] = amp * Math.sin(i / 7);
  return a;
}

test('short audio is one window', () => {
  const w = planWindows(tone(12));
  assert.equal(w.length, 1);
  assert.deepEqual([w[0].start, w[0].end, w[0].silent], [0, 12, false]);
});

test('long audio is cut at the quietest point, every window at most 30 s', () => {
  const a = tone(75);
  a.fill(0, Math.round(24 * SAMPLE_RATE), Math.round(24.4 * SAMPLE_RATE)); // a pause at 24 s
  const w = planWindows(a);
  assert.ok(w.length >= 3);
  assert.ok(w.every(x => x.end - x.start <= 30.001), JSON.stringify(w));
  assert.ok(Math.abs(w[0].end - 24.2) < 0.3, `first cut near the pause, got ${w[0].end}`);
  assert.equal(w.at(-1).end, 75);
  for (let i = 1; i < w.length; i++) assert.equal(w[i].start, w[i - 1].end);
});

test('silent windows are flagged', () => {
  const a = new Float32Array(40 * SAMPLE_RATE);
  a.set(tone(10), 0);
  const w = planWindows(a);
  assert.equal(w[0].silent, false);
  assert.equal(w.at(-1).silent, true);
});

test('noise and stock phrases are dropped, times become absolute', () => {
  const rows = chunksToRows([
    { start: 0, end: 2, text: ' ♪♪ ' },
    { start: 2, end: 5.5, text: ' Bajo la luna te esperé ' },
    { start: 6, end: null, text: 'Thank you for watching!' },
    { start: 7, end: null, text: 'con el corazón en la mano' }
  ], 30, 58);
  assert.deepEqual(rows.map(r => [r.start, r.end, r.orig]), [
    [32, 35.5, 'Bajo la luna te esperé'],
    [37, 58, 'con el corazón en la mano']
  ]);
  assert.ok(isNoise('[Music]'));
  assert.ok(!isNoise('Music is my home'));
});

test('a repeated line inside a repetition loop is merged', () => {
  const rows = chunksToRows([
    { start: 0, end: 2, text: 'la la la' },
    { start: 2.1, end: 4, text: 'la la la' }
  ], 0, 10);
  assert.equal(rows.length, 1);
  assert.equal(rows[0].end, 4);
});

test('translations attach to the row they overlap most', () => {
  const rows = [
    { start: 0, end: 4, orig: 'uno', en: '', note: '' },
    { start: 4, end: 9, orig: 'dos', en: '', note: '' }
  ];
  alignTranslations(rows, [
    { start: 0.2, end: 3.8, text: 'one' },
    { start: 3.9, end: 6, text: 'two,' },
    { start: 6, end: 9, text: 'still two' },
    { start: 20, end: 21, text: 'far away' }
  ]);
  assert.equal(rows[0].en, 'one');
  assert.equal(rows[1].en, 'two, still two far away');
});

test('timings are repaired and never overlap', () => {
  const rows = fixTimings([
    { start: 5, end: 4, orig: 'short line' },
    { start: 0, end: 7, orig: 'first' },
    { start: 6, end: 9, orig: 'last' }
  ], 8);
  assert.deepEqual(rows.map(r => r.orig), ['first', 'short line', 'last']);
  assert.ok(rows.every((r, i) => r.end > r.start && (i === rows.length - 1 || r.end <= rows[i + 1].start)));
  assert.equal(rows.at(-1).end, 8);
});

test('time formats', () => {
  assert.equal(fmtStamp(3725.042), '01:02:05,042');
  assert.equal(fmtStamp(1.5, '.'), '00:00:01.500');
  assert.equal(fmtClock(192.9), '3:12');
});

test('SRT and VTT output', () => {
  const rows = [
    { start: 1, end: 3.25, orig: 'Bajo la luna', en: 'Under the moon' },
    { start: 3.25, end: 5, orig: 'Hello', en: 'Hello' }
  ];
  assert.equal(buildSrt(rows, 'both'),
    '1\n00:00:01,000 --> 00:00:03,250\nBajo la luna\nUnder the moon\n\n2\n00:00:03,250 --> 00:00:05,000\nHello\n');
  assert.equal(buildSrt(rows, 'en').split('\n')[2], 'Under the moon');
  assert.ok(buildVtt(rows, 'orig').startsWith('WEBVTT\n\n00:00:01.000 --> 00:00:03.250\nBajo la luna\n'));
});

test('active row lookup and lyric sheet', () => {
  const rows = [{ start: 0, end: 2, orig: 'a', en: 'A' }, { start: 2, end: 4, orig: 'b', en: 'b', note: 'n' }];
  assert.equal(activeRow(rows, 1), 0);
  assert.equal(activeRow(rows, 3.9), 1);
  assert.equal(activeRow(rows, 9), -1);
  assert.equal(rowsToText(rows), '[0:00] a\n    A\n[0:02] b\n    (n)');
});
