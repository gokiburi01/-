/* MemoFit presentation layer. Exercise recognition and scoring stay in app.js. */
(() => {
  'use strict';
  const root = document.documentElement;
  const reduceMotion = window.matchMedia('(prefers-reduced-motion: reduce)');
  const STORAGE_KEY = 'memofit.records.v1';
  const MOTION_KEY = 'memofit.motion.v1';
  const english = { grip: 'Hand grip', highKnee: 'High knees', squat: 'Squat', jump: 'Jump' };
  const pageMeta = {
    homeScreen: ['01', 'MOVE. REMEMBER. REPEAT.'],
    genderScreen: ['02', 'YOUR PROFILE / GENDER'],
    ageScreen: ['03', 'YOUR PROFILE / AGE GROUP'],
    setupScreen: ['04', 'READY FOR YOUR FIRST TEST'],
    countdownScreen: ['05', 'TAKE A BREATH. FIND YOUR FOCUS.'],
    memoryScreen: ['06', 'REMEMBER THE NUMBERS'],
    answerScreen: ['07', 'ONE DIGIT AT A TIME'],
    beforeMemoryResultScreen: ['08', 'NEXT / MOVE YOUR BODY'],
    resultScreen: ['09', 'A LITTLE MOVEMENT. A NEW DISCOVERY.'],
    recordsScreen: ['10', 'YOUR TRAINING JOURNEY']
  };
  let motionPreference;
  let records = [];
  let storageAvailable = true;
  let sessionSaved = false;
  let introResolve = null;
  let introTimeouts = [];
  let toastTimeout;
  let lastCount = -1;
  let lastAchieved = false;
  let currentScreen = null;
  let inputWasComposing = false;
  try {
    motionPreference = localStorage.getItem(MOTION_KEY);
    const stored = JSON.parse(localStorage.getItem(STORAGE_KEY) || '[]');
    if (Array.isArray(stored)) records = stored.filter(validRecord).slice(0, 100);
  } catch { storageAvailable = false; }

  function validRecord(row) {
    return row && typeof row.date === 'string' && Number.isFinite(Date.parse(row.date)) &&
      ['before', 'after', 'squat', 'jump', 'kcal', 'duration'].every(key => Number.isFinite(row[key])) &&
      row.before >= 0 && row.before <= 100 && row.after >= 0 && row.after <= 100;
  }
  function motionOn() { return root.dataset.motion !== 'off'; }
  function applyMotion() {
    const enabled = motionPreference ? motionPreference === 'on' : !reduceMotion.matches;
    root.dataset.motion = enabled ? 'on' : 'off';
    $('motionToggle').setAttribute('aria-pressed', String(enabled));
    $('motionToggle').setAttribute('aria-label', enabled ? 'アニメーションをオフにする' : 'アニメーションをオンにする');
    $('motionState').textContent = enabled ? 'ON' : 'OFF';
  }
  applyMotion();
  reduceMotion.addEventListener('change', () => { if (!motionPreference) applyMotion(); });
  $('motionToggle').addEventListener('click', () => {
    motionPreference = motionOn() ? 'off' : 'on'; applyMotion();
    try { localStorage.setItem(MOTION_KEY, motionPreference); } catch { /* Keep the session preference. */ }
  });

  function animate(element, keyframes, options = {}) {
    if (!motionOn() || !element?.animate) return;
    return element.animate(keyframes, { duration: 450, easing: 'cubic-bezier(.22,1,.36,1)', ...options });
  }
  function tick(element) {
    animate(element, [{ opacity: .3, transform: 'translateY(10px) scale(.94)' }, { opacity: 1, transform: 'translateY(0) scale(1)' }]);
  }
  function numberTo(element, target, { prefix = '', suffix = '', duration = 1100 } = {}) {
    const token = Symbol('number'); element._numberToken = token;
    if (!motionOn()) { element.textContent = prefix + target + suffix; return; }
    const started = performance.now();
    function frame(now) {
      if (element._numberToken !== token) return;
      const t = Math.min(1, (now - started) / duration);
      const value = Math.round(target * (1 - Math.pow(1 - t, 3)));
      element.textContent = prefix + value + suffix;
      if (t < 1) requestAnimationFrame(frame);
    }
    requestAnimationFrame(frame);
  }
  function toast(message) {
    $('toast').textContent = message; $('toast').classList.add('visible');
    clearTimeout(toastTimeout); toastTimeout = setTimeout(() => $('toast').classList.remove('visible'), 3000);
  }
  function screenChanged(screen) {
    const oldScreen = currentScreen;
    currentScreen = screen;
    const immersive = screen === trainingScreen || screen === loadingScreen;
    $('siteHeader').classList.toggle('hidden', immersive);
    $('siteFooter').classList.toggle('hidden', immersive);
    root.dataset.screen = screen.id;
    screen.classList.remove('entering');
    const reveals = screen.querySelectorAll('.reveal');
    reveals.forEach((item, index) => item.style.setProperty('--delay', `${index * 85}ms`));
    // Reflow once at navigation; never in the pose loop.
    void screen.offsetWidth;
    screen.classList.add('entering');
    if (oldScreen && oldScreen !== screen && motionOn() && screen !== memoryScreen && screen !== answerScreen) {
      animate(screen, [{ opacity: .3 }, { opacity: 1 }], { duration: 350 });
    }
    if (pageMeta[screen.id]) {
      $('pageIndex').textContent = pageMeta[screen.id][0];
      $('footerCaption').textContent = pageMeta[screen.id][1];
    }
    const profilePages = new Set(['homeScreen', 'genderScreen', 'ageScreen', 'setupScreen']);
    if (oldScreen && oldScreen !== screen && profilePages.has(oldScreen.id) && profilePages.has(screen.id)) {
      animate($('transitionCurtain'), [
        { transform: 'translateX(-110%) skewX(-10deg)', offset: 0 },
        { transform: 'translateX(0) skewX(-10deg)', offset: .42 },
        { transform: 'translateX(110%) skewX(-10deg)', offset: 1 }
      ], { duration: 600 });
    }
    document.querySelectorAll('[data-phase-label]').forEach(item => {
      item.textContent = phase === 'before' ? 'BEFORE TRAINING / 運動前' : 'AFTER TRAINING / 運動後';
    });
    window.scrollTo({ top: 0, behavior: 'instant' });
    if (screen === countdownScreen) {
      if (phase === 'before') sessionSaved = false;
      $('countdownTitle').innerHTML = phase === 'before' ? '数字を覚える、<br>準備をしましょう。' : '運動のあとは、<br>もう一度チャレンジ。';
      $('countdownRing').style.strokeDashoffset = '0';
    }
    if (screen === answerScreen) renderAnswer();
    if (screen === beforeMemoryResultScreen) numberTo(beforeRate, beforeScore);
    if (screen === loadingScreen) {
      $('cancelLoadingBtn').textContent = '準備を中止して戻る ↙';
      $('loadingScreen').classList.remove('loadFailed');
    }
    if (screen === resultScreen) {
      numberTo(beforeRateResult, beforeScore, { suffix: '%' });
      numberTo(afterRateResult, afterScore, { suffix: '%' });
      numberTo(improveRate, improveScore, { prefix: improveScore >= 0 ? '+' : '' });
      $('beforeBar').style.transform = 'scaleX(0)'; $('afterBar').style.transform = 'scaleX(0)';
      requestAnimationFrame(() => requestAnimationFrame(() => {
        $('beforeBar').style.transform = `scaleX(${beforeScore / 100})`;
        $('afterBar').style.transform = `scaleX(${afterScore / 100})`;
      }));
    }
    if (screen === $('recordsScreen')) renderRecords();
    // Keep keyboard focus on the active screen after a page change.
    const heading = screen.querySelector('h1');
    if (heading && screen !== answerScreen && screen !== trainingScreen) {
      heading.tabIndex = -1; heading.focus({ preventScroll: true });
    }
  }

  document.querySelectorAll('[data-go]').forEach(button => button.addEventListener('click', () => showScreen($(button.dataset.go))));
  $('beginBtn').addEventListener('click', () => showScreen($('genderScreen')));
  $('brandHome').addEventListener('click', restartApp);
  $('viewRecordsBtn').addEventListener('click', () => showScreen($('recordsScreen')));
  $('resultRecordsBtn').addEventListener('click', () => showScreen($('recordsScreen')));
  $('cancelLoadingBtn').addEventListener('click', cancelPreparation);
  document.querySelectorAll('[data-gender]').forEach(button => button.addEventListener('click', () => {
    genderInput.value = button.dataset.gender;
    document.querySelectorAll('[data-gender]').forEach(choice => choice.setAttribute('aria-pressed', String(choice === button)));
    animate(button, [{ transform: 'scale(.96)' }, { transform: 'scale(1)' }]);
  }));
  function changeAge(delta) {
    const next = Math.min(ageGroupInput.options.length - 1, Math.max(0, ageGroupInput.selectedIndex + delta));
    ageGroupInput.selectedIndex = next;
    const value = ageGroupInput.value;
    $('ageValue').replaceChildren();
    if (value === 'other') {
      $('ageValue').textContent = 'その他';
      $('ageValue').style.fontSize = 'clamp(35px, 6vw, 70px)';
    } else {
      $('ageValue').style.fontSize = '';
      const [from, to] = value.split('-');
      $('ageValue').append(document.createTextNode(from));
      const dash = document.createElement('span'); dash.textContent = '〜'; $('ageValue').append(dash, document.createTextNode(to));
      const unit = document.createElement('small'); unit.textContent = '代'; $('ageValue').append(unit);
    }
    $('ageDown').disabled = next === 0; $('ageUp').disabled = next === ageGroupInput.options.length - 1;
    tick($('ageValue'));
  }
  $('ageDown').addEventListener('click', () => changeAge(-1));
  $('ageUp').addEventListener('click', () => changeAge(1));
  changeAge(0);

  function renderDigits(digits) {
    memoryDigits.replaceChildren();
    memoryDigits.setAttribute('aria-label', digits.split('').join(' '));
    Array.from(digits).forEach((digit, index) => {
      const circle = document.createElement('span'); circle.className = 'digitCircle';
      circle.style.setProperty('--i', index); circle.textContent = digit;
      circle.dataset.index = String(index + 1).padStart(2, '0');
      memoryDigits.append(circle);
    });
  }
  function renderAnswer() {
    const value = memoryAnswerInput.value;
    const slots = $('answerSlots');
    if (!slots.children.length) {
      for (let i = 0; i < MEMORY_LENGTH; i += 1) {
        const circle = document.createElement('span'); circle.className = 'digitCircle';
        circle.dataset.index = String(i + 1).padStart(2, '0'); circle.style.setProperty('--i', i);
        slots.append(circle);
      }
    }
    Array.from(slots.children).forEach((circle, i) => {
      const text = value[i] ?? '?';
      if (circle.textContent !== text) {
        circle.textContent = text;
        if (value[i]) animate(circle, [{ transform: 'scale(.8)' }, { transform: 'scale(1)' }], { duration: 280 });
      }
      circle.classList.toggle('filled', i < value.length);
      circle.classList.toggle('active', i === value.length && document.activeElement === memoryAnswerInput);
    });
    $('answerLength').textContent = value.length;
  }
  function sanitizeAnswer() {
    if (inputWasComposing) return;
    // Full-width numbers from a Japanese keyboard are accepted too.
    memoryAnswerInput.value = memoryAnswerInput.value.replace(/[０-９]/g, char => String.fromCharCode(char.charCodeAt(0) - 0xFEE0)).replace(/\D/g, '').slice(0, MEMORY_LENGTH);
    renderAnswer();
  }
  memoryAnswerInput.placeholder = '数字を順番に入力';
  memoryAnswerInput.addEventListener('input', sanitizeAnswer);
  memoryAnswerInput.addEventListener('compositionstart', () => { inputWasComposing = true; });
  memoryAnswerInput.addEventListener('compositionend', () => { inputWasComposing = false; sanitizeAnswer(); });
  memoryAnswerInput.addEventListener('focus', renderAnswer);
  memoryAnswerInput.addEventListener('blur', renderAnswer);
  function countdownTick(seconds) {
    $('countdownRing').style.strokeDashoffset = String(578 * (1 - Math.max(0, seconds) / INTRO_COUNTDOWN));
    tick(countdownNumber);
  }
  function memoryTick(seconds) {
    $('memoryProgressFill').style.transform = `scaleX(${Math.max(0, seconds) / MEMORY_TIME})`;
    memoryTimer.parentElement.classList.toggle('urgent', seconds <= 5);
    tick(memoryTimer);
  }

  function cancelIntro() {
    introTimeouts.forEach(clearTimeout); introTimeouts = [];
    $('exerciseIntro').classList.add('hidden');
    if (introResolve) { const resolve = introResolve; introResolve = null; resolve(false); }
  }
  function introduceExercise(exercise) {
    cancelIntro();
    const overlay = $('exerciseIntro');
    $('introExerciseName').textContent = exercise.name;
    $('introExerciseEnglish').textContent = english[exercise.type];
    $('exerciseIntroNumber').textContent = '3'; overlay.classList.remove('hidden');
    tick($('exerciseIntroNumber'));
    return new Promise(resolve => {
      introResolve = resolve;
      [2, 1].forEach((number, i) => introTimeouts.push(setTimeout(() => {
        $('exerciseIntroNumber').textContent = number; tick($('exerciseIntroNumber'));
      }, (i + 1) * 1000)));
      introTimeouts.push(setTimeout(() => {
        overlay.classList.add('hidden'); introTimeouts = []; introResolve = null; resolve(true);
      }, 3000));
    });
  }
  function exerciseChanged(exercise) {
    $('exerciseEnglish').textContent = english[exercise.type];
    $('trainingMascot').dataset.exercise = exercise.type;
    lastCount = -1; lastAchieved = false;
    showWarning('');
    excellentBadge.style.display = 'none';
    trainingTick(); updateLive();
  }
  function trainingTick() {
    $('setSeconds').textContent = Math.max(0, remainExerciseTime);
    $('setTimeFill').style.transform = `scaleX(${Math.max(0, remainExerciseTime) / EXERCISE_TIME})`;
    const dots = $('setDots');
    if (dots.children.length !== TOTAL_EXERCISES) {
      dots.replaceChildren(...Array.from({ length: TOTAL_EXERCISES }, () => document.createElement('span')));
    }
    Array.from(dots.children).forEach((dot, i) => {
      dot.classList.toggle('done', i < completedExercises); dot.classList.toggle('current', i === completedExercises);
    });
    progressText.textContent = `${completedExercises + 1} / ${TOTAL_EXERCISES} セット`;
  }
  function updateLive() {
    if (currentScreen !== trainingScreen) return;
    const count = currentExerciseCount();
    if (count !== lastCount) {
      $('currentReps').textContent = count;
      if (count > lastCount && lastCount >= 0) tick($('currentReps'));
      lastCount = count;
    }
    if (count >= GOAL_REPS && !lastAchieved) { lastAchieved = true; excellent(); celebrate(); }
  }
  function excellent() {
    clearTimeout(excellentTimer);
    excellentBadge.style.display = 'block';
    animate(excellentBadge, [
      { opacity: 0, transform: 'translate(-50%,-35%) rotate(-8deg) scale(.75)' },
      { opacity: 1, transform: 'translate(-50%,-50%) rotate(-5deg) scale(1)' }
    ], { duration: 600 });
    excellentTimer = setTimeout(() => { excellentBadge.style.display = 'none'; }, 1500);
  }
  function celebrate() {
    if (!motionOn()) return;
    const container = $('celebration'); container.replaceChildren();
    for (let i = 0; i < 24; i += 1) {
      const piece = document.createElement('i'); piece.className = 'confetti';
      piece.style.left = `${Math.random() * 100}%`;
      piece.style.setProperty('--drift', `${(Math.random() - .5) * 250}px`);
      piece.style.setProperty('--spin', `${Math.random() * 700 - 350}deg`);
      piece.style.animationDelay = `${Math.random() * .3}s`;
      piece.style.background = ['#fff', '#a3a3a3', '#b9dfef', '#c9e8d6'][i % 4];
      container.append(piece);
      piece.addEventListener('animationend', () => piece.remove(), { once: true });
    }
  }
  function loadingFailed() {
    $('cancelLoadingBtn').textContent = 'テスト結果に戻って、もう一度試す ↙';
    $('loadingScreen').classList.add('loadFailed');
  }
  function optionalNumber(input) {
    const value = Number(input.value);
    return input.value && input.validity.valid && Number.isFinite(value) ? value : '';
  }
  function saveResult() {
    if (sessionSaved) return;
    sessionSaved = true;
    const row = {
      date: new Date().toISOString(), before: beforeScore, after: afterScore,
      beforeCorrect: beforeCorrectCount, afterCorrect: afterCorrectCount,
      squat: squatCount, jump: jumpCount, grip: gripCount, highKnee: highKneeCount,
      kcal: Number(calorie.toFixed(1)), duration: elapsedTraining,
      gender: genderInput.value, ageGroup: ageGroupInput.value,
      height: optionalNumber(heightInput), weight: optionalNumber(weightInput)
    };
    records.unshift(row); records = records.slice(0, 100);
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(records)); storageAvailable = true;
      $('recordStatus').textContent = '今回の記録を、このブラウザに保存しました。';
    } catch {
      storageAvailable = false;
      $('recordStatus').textContent = 'ブラウザに保存できませんでした。記録画面からCSVを書き出せます。';
    }
  }
  function renderRecords() {
    const container = $('recordsList'); container.replaceChildren();
    $('exportRecordsBtn').disabled = records.length === 0;
    $('recordsNote').textContent = storageAvailable
      ? '記録は、この端末のこのブラウザに保存されます（最新100件）。'
      : 'ブラウザ保存が利用できません。画面を閉じる前にCSVを書き出してください。';
    if (!records.length) {
      const empty = document.createElement('div'); empty.className = 'emptyRecords';
      empty.innerHTML = '<span class="emptyIcon" aria-hidden="true">01 ↗</span><h2>はじめの一歩を、ここから。</h2><p>トレーニングを終えると、記録がここに並びます。<br>まずは、ひとつ試してみましょう。</p>';
      container.append(empty); return;
    }
    records.forEach((row, index) => {
      const line = document.createElement('article'); line.className = 'recordRow'; line.style.setProperty('--i', Math.min(index, 8));
      const date = new Date(row.date);
      const dateCell = document.createElement('div'); dateCell.className = 'recordDate';
      dateCell.textContent = date.toLocaleDateString('ja-JP', { year: 'numeric', month: '2-digit', day: '2-digit' });
      const time = document.createElement('small');
      time.textContent = `${date.toLocaleTimeString('ja-JP', { hour: '2-digit', minute: '2-digit' })} / ${Math.floor(row.duration / 60)}分${row.duration % 60}秒`;
      dateCell.append(time); line.append(dateCell);
      for (const [label, value] of [['BEFORE / 運動前', row.before], ['AFTER / 運動後', row.after]]) {
        const cell = document.createElement('div'); cell.className = 'recordScore';
        const title = document.createElement('span'); title.textContent = label;
        const score = document.createElement('b'); score.textContent = value + '%'; cell.append(title, score); line.append(cell);
      }
      const difference = row.after - row.before;
      const change = document.createElement('div'); change.className = 'recordChange';
      const label = document.createElement('span'); label.textContent = 'CHANGE / 変化';
      const value = document.createElement('b'); value.textContent = (difference >= 0 ? '+' : '') + difference;
      const unit = document.createElement('small'); unit.textContent = 'pt'; change.append(label, value, unit); line.append(change);
      container.append(line);
    });
  }
  $('exportRecordsBtn').addEventListener('click', () => {
    if (!records.length) return;
    const headers = ['日時', '運動前正解数', '運動後正解数', '運動前正答率(%)', '運動後正答率(%)', '正答率の差(pt)', '運動時間(秒)', 'グーパー回数', 'もも上げ回数', 'スクワット回数', 'ジャンプ回数', '推定消費kcal', '性別', '年齢層', '身長(cm)', '体重(kg)'];
    const quote = value => '"' + String(value ?? '').replace(/"/g, '""') + '"';
    const rows = records.map(row => [row.date, row.beforeCorrect, row.afterCorrect, row.before, row.after, row.after - row.before, row.duration, row.grip, row.highKnee, row.squat, row.jump, row.kcal, row.gender, row.ageGroup, row.height, row.weight]);
    const csv = '\uFEFF' + [headers, ...rows].map(row => row.map(quote).join(',')).join('\r\n');
    const url = URL.createObjectURL(new Blob([csv], { type: 'text/csv;charset=utf-8' }));
    const link = document.createElement('a'); link.href = url; link.download = 'MemoFit-records.csv';
    document.body.append(link); link.click(); link.remove();
    setTimeout(() => URL.revokeObjectURL(url), 1000); toast('CSVを書き出しました。');
  });

  document.addEventListener('click', event => {
    const button = event.target.closest('.slashBtn');
    if (!button || !motionOn()) return;
    const rect = button.getBoundingClientRect();
    const ripple = document.createElement('span'); ripple.className = 'ripple';
    ripple.style.left = `${(event.clientX || rect.left + rect.width / 2) - rect.left - 10}px`;
    ripple.style.top = `${(event.clientY || rect.top + rect.height / 2) - rect.top - 10}px`;
    button.append(ripple); ripple.addEventListener('animationend', () => ripple.remove(), { once: true });
  });
  // The SVG is embedded so it also works when index.html is opened from a file.
  const mascotSVG = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 260 280"><ellipse class="mascotShadow" cx="130" cy="259" rx="71" ry="10" fill="#000" opacity=".18"/><g class="mascotLegs" fill="#dddcd8"><rect class="legLeft" x="85" y="219" width="31" height="33" rx="13"/><rect class="legRight" x="144" y="219" width="31" height="33" rx="13"/></g><g class="mascotBody"><path d="M97 186h66l10 39q-4 18-19 18h-48q-15 0-19-18Z" fill="#eae9e5"/><rect x="94" y="192" width="72" height="23" rx="7" fill="#272727"/><g class="mascotArm armLeft"><path d="m86 181-33 23" stroke="#3b3b39" stroke-width="14" stroke-linecap="round"/><circle cx="46" cy="210" r="21" fill="#eae9e5"/></g><g class="mascotArm armRight"><path d="m174 181 32 12" stroke="#3b3b39" stroke-width="14" stroke-linecap="round"/><circle cx="213" cy="190" r="21" fill="#eae9e5"/></g><g class="mascotHead"><circle cx="130" cy="43" r="24" fill="#d5d5d0"/><rect x="27" y="104" width="34" height="56" rx="12" fill="#d5d5d0"/><rect x="199" y="104" width="34" height="56" rx="12" fill="#d5d5d0"/><path d="M130 38c57 0 92 48 92 108 0 45-37 62-92 62s-92-17-92-62c0-60 35-108 92-108Z" fill="#eae9e5"/><path d="M130 61c42 0 70 39 70 84 0 29-28 41-70 41s-70-12-70-41c0-45 28-84 70-84Z" fill="#232323"/><g class="mascotEyes" stroke="#eae9e5" fill="none" stroke-width="4" stroke-linecap="round"><path d="M91 117q14 1 19 14-14-5-25 0m84-14q-14 1-19 14 14-5 25 0"/></g><path d="M119 139h22q2 25-11 25t-11-25Z" fill="#eae9e5"/></g></g></svg>`;
  document.querySelectorAll('[data-mascot]').forEach(container => { container.innerHTML = mascotSVG; });
  window.MemoUI = { screenChanged, renderDigits, countdownTick, memoryTick, introduceExercise, cancelIntro, exerciseChanged, trainingTick, updateLive, excellent, loadingFailed, saveResult };
  showScreen($('homeScreen'));
})();
