// Система визуального распознавания в технологичном стиле
class VisionSystem {
  constructor(canvasId) {
    this.canvas = document.getElementById(canvasId);
    this.ctx = this.canvas.getContext('2d');
    this.overlay = document.createElement('div');
    this.overlay.className = 'vision-overlay';
    this.canvas.parentElement.appendChild(this.overlay);
    
    // Состояние
    this.mode = 'SEGMENT';
    this.fps = 0;
    this.detectedObjects = [];
    this.handData = null;
    this.trackingData = null;
    
    // Инициализация элементов UI
    this.initUI();
    
    // Подключение WebSocket
    this.connectVisionWS();
  }
  
  initUI() {
    // Индикатор состояния
    this.statusElement = this.createStatusIndicator();
    this.overlay.appendChild(this.statusElement);
    
    // Индикаторы прогресса
    this.progressElement = this.createProgressBars();
    this.overlay.appendChild(this.progressElement);
    
    // Панель данных
    this.dataPanel = this.createDataPanel();
    this.overlay.appendChild(this.dataPanel);
  }
  
  createStatusIndicator() {
    const status = document.createElement('div');
    status.className = 'status-indicator';
    status.innerHTML = `
      <div class="status-main">Система готова <span class="status-sub">System Ready</span></div>
      <div class="status-sub">Ожидание цели Waiting for Target</div>
    `;
    return status;
  }
  
  createProgressBars() {
    const container = document.createElement('div');
    container.className = 'progress-container';
    container.innerHTML = `
      <div class="progress-item">
        <div class="progress-label">
          <span class="progress-label-text">Выравнивание <span class="progress-label-sub">Alignment</span></span>
          <span class="progress-value">0%</span>
        </div>
        <div class="progress-bar">
          <div class="progress-fill" id="align-progress" style="width: 0%"></div>
        </div>
      </div>
      <div class="progress-item">
        <div class="progress-label">
          <span class="progress-label-text">Соответствие расстояния <span class="progress-label-sub">Distance Match</span></span>
          <span class="progress-value">0%</span>
        </div>
        <div class="progress-bar">
          <div class="progress-fill" id="distance-progress" style="width: 0%"></div>
        </div>
      </div>
    `;
    return container;
  }
  
  createDataPanel() {
    const panel = document.createElement('div');
    panel.className = 'data-panel';
    panel.innerHTML = `
      <div class="data-item">
        <span class="data-label">FPS</span>
        <span class="data-value" id="fps-value">--</span>
      </div>
      <div class="data-item">
        <span class="data-label">Режим Mode</span>
        <span class="data-value" id="mode-value">Обнаружение</span>
      </div>
      <div class="data-item">
        <span class="data-label">Объектов Objects</span>
        <span class="data-value" id="objects-value">0</span>
      </div>
      <div class="data-item">
        <span class="data-label">Оценка захвата Grasp</span>
        <span class="data-value" id="grasp-value">0.00</span>
      </div>
    `;
    return panel;
  }
  
  connectVisionWS() {
    const proto = location.protocol === 'https:' ? 'wss' : 'ws';
    this.ws = new WebSocket(`${proto}://${location.host}/ws/viewer`);  // изменено на /ws/viewer
    
    this.ws.onopen = () => {
      console.log('[Vision] WebSocket connected');
      // ... rest of the code
    };
    
    this.ws.onmessage = (event) => {
      // Обработка двоичных данных изображения
      if (event.data instanceof Blob) {
        // Создаём URL изображения и отображаем
        const url = URL.createObjectURL(event.data);
        const img = new Image();
        img.onload = () => {
          this.ctx.drawImage(img, 0, 0, this.canvas.width, this.canvas.height);
          URL.revokeObjectURL(url);
        };
        img.src = url;
      }
    };
    
    this.ws.onerror = () => {
      console.error('Vision WebSocket error');
    };
  }
  
  updateVisualization(data) {
    // Обновление состояния
    this.mode = data.mode || 'SEGMENT';
    this.fps = data.fps || 0;
    
    // Обновление UI
    this.updateStatus(data);
    this.updateProgress(data);
    this.updateDataPanel(data);
    
    // Отрисовка визуализации
    if (data.frame) {
      this.drawFrame(data.frame);
    }
    
    if (data.hand) {
      this.drawHand(data.hand);
    }
    
    if (data.objects) {
      this.drawObjects(data.objects);
    }
    
    if (data.tracking) {
      this.drawTracking(data.tracking);
    }
  }
  
  updateStatus(data) {
    const statusMain = this.statusElement.querySelector('.status-main');
    const statusSub = this.statusElement.querySelector('.status-sub:last-child');
    
    switch(this.mode) {
      case 'SEGMENT':
        statusMain.innerHTML = 'Обнаружение цели <span class="status-sub">Detecting</span>';
        statusSub.textContent = data.message || 'Сканирование окружения Scanning Environment';
        break;
      case 'FLASH':
        statusMain.innerHTML = 'Захват цели <span class="status-sub">Locking</span>';
        statusSub.textContent = 'Подготовка к отслеживанию Preparing to Track';
        break;
      case 'TRACK':
        statusMain.innerHTML = 'Отслеживание <span class="status-sub">Tracking</span>';
        statusSub.textContent = 'Удерживайте наведение Maintain Alignment';
        break;
    }
  }
  
  updateProgress(data) {
    if (data.alignScore !== undefined) {
      const alignPercent = Math.round(data.alignScore * 100);
      document.getElementById('align-progress').style.width = `${alignPercent}%`;
      this.progressElement.querySelector('.progress-value').textContent = `${alignPercent}%`;
    }
    
    if (data.distanceScore !== undefined) {
      const distPercent = Math.round(data.distanceScore * 100);
      document.getElementById('distance-progress').style.width = `${distPercent}%`;
      this.progressElement.querySelectorAll('.progress-value')[1].textContent = `${distPercent}%`;
    }
  }
  
  updateDataPanel(data) {
    document.getElementById('fps-value').textContent = Math.round(this.fps);
    document.getElementById('mode-value').textContent = this.getModeText(this.mode);
    document.getElementById('objects-value').textContent = data.objectCount || 0;
    document.getElementById('grasp-value').textContent = (data.graspScore || 0).toFixed(2);
  }
  
  getModeText(mode) {
    const modeMap = {
      'SEGMENT': 'Обнаружение Detect',
      'FLASH': 'Захват Lock',
      'TRACK': 'Отслеживание Track'
    };
    return modeMap[mode] || mode;
  }
  
  drawFrame(frameData) {
    // Отрисовка базового изображения
    const img = new Image();
    img.onload = () => {
      this.canvas.width = img.width;
      this.canvas.height = img.height;
      this.ctx.drawImage(img, 0, 0);
    };
    img.src = 'data:image/jpeg;base64,' + frameData;
  }
  
  drawHand(handData) {
    // Отрисовка скелета руки с помощью SVG
    const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
    svg.style.position = 'absolute';
    svg.style.top = '0';
    svg.style.left = '0';
    svg.style.width = '100%';
    svg.style.height = '100%';
    svg.style.pointerEvents = 'none';
    
    // Отрисовка соединительных линий
    handData.connections.forEach(conn => {
      const line = document.createElementNS('http://www.w3.org/2000/svg', 'line');
      line.setAttribute('x1', conn.start.x);
      line.setAttribute('y1', conn.start.y);
      line.setAttribute('x2', conn.end.x);
      line.setAttribute('y2', conn.end.y);
      line.setAttribute('class', 'hand-skeleton');
      svg.appendChild(line);
    });
    
    // Отрисовка суставов
    handData.landmarks.forEach(point => {
      const circle = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
      circle.setAttribute('cx', point.x);
      circle.setAttribute('cy', point.y);
      circle.setAttribute('r', '3');
      circle.setAttribute('class', 'hand-joint');
      svg.appendChild(circle);
    });
    
    // Добавление на оверлей
    const oldSvg = this.overlay.querySelector('svg');
    if (oldSvg) oldSvg.remove();
    this.overlay.appendChild(svg);
  }
  
  drawObjects(objects) {
    // Отрисовка обнаруженных объектов
    objects.forEach((obj, index) => {
      if (obj.isTarget) {
        // Целевой объект — особым стилем
        this.drawTargetObject(obj);
      } else {
        // Прочие объекты — обычным стилем
        this.drawNormalObject(obj);
      }
    });
  }
  
  drawTargetObject(obj) {
    // Создание эффекта захвата цели
    const target = document.createElement('div');
    target.className = 'target-lock';
    target.style.position = 'absolute';
    target.style.left = `${obj.x}px`;
    target.style.top = `${obj.y}px`;
    target.style.width = `${obj.width}px`;
    target.style.height = `${obj.height}px`;
    
    // Добавление анимации захвата
    const svg = `
      <svg width="${obj.width}" height="${obj.height}" style="position: absolute; top: 0; left: 0;">
        <rect x="2" y="2" width="${obj.width-4}" height="${obj.height-4}" 
              class="target-lock" rx="8" ry="8"/>
      </svg>
    `;
    target.innerHTML = svg;
    
    this.overlay.appendChild(target);
  }
}

// Инициализация
document.addEventListener('DOMContentLoaded', () => {
  const visionSystem = new VisionSystem('vision-canvas');
}); 