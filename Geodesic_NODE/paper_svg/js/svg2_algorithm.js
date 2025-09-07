// SVG 2: GEOSPEC Algorithm - Learning Where to Sense
function createSVG2(data) {
    const width = 1200;
    const height = 600;
    const margin = { top: 20, right: 20, bottom: 20, left: 20 };
    
    // Create main SVG
    const svg = d3.select('#svg2-container')
        .append('svg')
        .attr('width', width)
        .attr('height', height)
        .attr('viewBox', `0 0 ${width} ${height}`);
    
    // Define common elements
    const defs = svg.append('defs');
    
    // Arrow marker for vectors
    defs.append('marker')
        .attr('id', 'arrowhead')
        .attr('markerWidth', 10)
        .attr('markerHeight', 7)
        .attr('refX', 9)
        .attr('refY', 3.5)
        .attr('orient', 'auto')
        .append('polygon')
        .attr('points', '0 0, 10 3.5, 0 7')
        .attr('fill', '#8b5cf6');
    
    // Panel dimensions
    const panelWidth = 300;
    const panelHeight = 600;
    
    // Create 4 panels
    createPanel1_GeodesicDistance(svg, 0, 0, panelWidth, panelHeight, data);
    createPanel2_NeuralArchitecture(svg, panelWidth, 0, panelWidth, panelHeight);
    createPanel3_ComputationalEfficiency(svg, panelWidth * 2, 0, panelWidth, panelHeight);
    createPanel4_ShootingResults(svg, panelWidth * 3, 0, panelWidth, panelHeight, data);
}

// Panel 1: Geodesic Distance
function createPanel1_GeodesicDistance(svg, x, y, width, height, data) {
    const panel = svg.append('g')
        .attr('transform', `translate(${x}, ${y})`);
    
    // Panel border
    panel.append('rect')
        .attr('width', width)
        .attr('height', height)
        .attr('class', 'panel-border')
        .attr('fill', 'white');
    
    // Title
    panel.append('text')
        .attr('x', width / 2)
        .attr('y', 30)
        .attr('text-anchor', 'middle')
        .attr('class', 'header-text')
        .text('Geodesic Distance');
    
    // Metric tensor heatmap
    const heatmapGroup = panel.append('g')
        .attr('transform', 'translate(30, 60)');
    
    const heatmapWidth = width - 60;
    const heatmapHeight = 180;
    
    drawMetricTensorHeatmap(heatmapGroup, heatmapWidth, heatmapHeight);
    
    // Christoffel symbols vector field
    const christoffelGroup = panel.append('g')
        .attr('transform', 'translate(30, 280)');
    
    drawChristoffelField(christoffelGroup, heatmapWidth, 150);
    
    // Hamiltonian contours
    const hamiltonianGroup = panel.append('g')
        .attr('transform', 'translate(30, 460)');
    
    drawHamiltonianContours(hamiltonianGroup, heatmapWidth, 120);
}

// Panel 2: Neural Architecture
function createPanel2_NeuralArchitecture(svg, x, y, width, height) {
    const panel = svg.append('g')
        .attr('transform', `translate(${x}, ${y})`);
    
    // Panel border
    panel.append('rect')
        .attr('width', width)
        .attr('height', height)
        .attr('class', 'panel-border')
        .attr('fill', 'white');
    
    // Title
    panel.append('text')
        .attr('x', width / 2)
        .attr('y', 30)
        .attr('text-anchor', 'middle')
        .attr('class', 'header-text')
        .text('Neural Architecture');
    
    // Metric Network
    const metricNetGroup = panel.append('g')
        .attr('transform', 'translate(50, 80)');
    
    drawMetricNetwork(metricNetGroup, width - 100);
    
    // Flow Network
    const flowNetGroup = panel.append('g')
        .attr('transform', 'translate(50, 350)');
    
    drawFlowNetwork(flowNetGroup, width - 100);
}

// Panel 3: Computational Efficiency (Combined)
function createPanel3_ComputationalEfficiency(svg, x, y, width, height) {
    const panel = svg.append('g')
        .attr('transform', `translate(${x}, ${y})`);
    
    // Panel border
    panel.append('rect')
        .attr('width', width)
        .attr('height', height)
        .attr('class', 'panel-border')
        .attr('fill', 'white');
    
    // Title
    panel.append('text')
        .attr('x', width / 2)
        .attr('y', 30)
        .attr('text-anchor', 'middle')
        .attr('class', 'header-text')
        .text('Computational Efficiency');
    
    // Top Half: Memory (Adjoint Method)
    const memoryGroup = panel.append('g')
        .attr('transform', 'translate(30, 60)');
    
    drawMemoryComparison(memoryGroup, width - 60, 250);
    
    // Bottom Half: Time (Christoffel Caching)
    const timeGroup = panel.append('g')
        .attr('transform', 'translate(30, 330)');
    
    drawTimeComparison(timeGroup, width - 60, 250);
}

// Panel 4: Shooting Algorithm & Results
function createPanel4_ShootingResults(svg, x, y, width, height, data) {
    const panel = svg.append('g')
        .attr('transform', `translate(${x}, ${y})`);
    
    // Panel border
    panel.append('rect')
        .attr('width', width)
        .attr('height', height)
        .attr('class', 'panel-border')
        .attr('fill', 'white');
    
    // Title
    panel.append('text')
        .attr('x', width / 2)
        .attr('y', 30)
        .attr('text-anchor', 'middle')
        .attr('class', 'header-text')
        .text('Shooting & Results');
    
    // Shooting algorithm visualization
    const shootingGroup = panel.append('g')
        .attr('transform', 'translate(30, 60)');
    
    drawShootingAlgorithm(shootingGroup, width - 60, 200);
    
    // Spectral reconstruction
    const spectraGroup = panel.append('g')
        .attr('transform', 'translate(30, 280)');
    
    drawSpectralReconstruction(spectraGroup, width - 60, 200, data);
    
    // Color mapping
    const colorGroup = panel.append('g')
        .attr('transform', 'translate(30, 500)');
    
    drawColorMapping(colorGroup, width - 60, 80);
}

// Helper functions for each visualization

function drawMetricTensorHeatmap(group, width, height) {
    // Generate heatmap data
    const resolution = 30;
    const data = generateMetricTensorData(width, height, resolution);
    
    // Color scale
    const colorScale = d3.scaleSequential()
        .domain([0, 1])
        .interpolator(d3.interpolateViridis);
    
    // Draw cells
    const cellWidth = width / resolution;
    const cellHeight = height / resolution;
    
    group.selectAll('.heatmap-cell')
        .data(data)
        .enter()
        .append('rect')
        .attr('class', 'heatmap-cell')
        .attr('x', d => d.x * cellWidth)
        .attr('y', d => d.y * cellHeight)
        .attr('width', cellWidth)
        .attr('height', cellHeight)
        .attr('fill', d => colorScale(d.value));
    
    // Add equation label
    group.append('text')
        .attr('x', width / 2)
        .attr('y', -10)
        .attr('text-anchor', 'middle')
        .attr('class', 'math-text')
        .text('g(c,λ)');
    
    // Add axis labels
    group.append('text')
        .attr('x', width / 2)
        .attr('y', height + 20)
        .attr('text-anchor', 'middle')
        .attr('class', 'label-text')
        .text('Wavelength (nm)');
    
    group.append('text')
        .attr('x', -10)
        .attr('y', height / 2)
        .attr('text-anchor', 'middle')
        .attr('transform', `rotate(-90, -10, ${height/2})`)
        .attr('class', 'label-text')
        .text('Concentration (ppb)');
}

function drawChristoffelField(group, width, height) {
    const vectors = generateChristoffelField(width, height, 15);
    
    // Color scale for magnitude
    const colorScale = d3.scaleSequential()
        .domain([0, 1])
        .interpolator(d3.interpolatePurples);
    
    vectors.forEach(v => {
        group.append('line')
            .attr('x1', v.x)
            .attr('y1', v.y)
            .attr('x2', v.x + v.dx)
            .attr('y2', v.y + v.dy)
            .attr('stroke', colorScale(v.magnitude))
            .attr('stroke-width', 1.5)
            .attr('marker-end', 'url(#arrowhead)');
    });
    
    // Add equation label
    group.append('text')
        .attr('x', width / 2)
        .attr('y', -10)
        .attr('text-anchor', 'middle')
        .attr('class', 'math-text')
        .text('Γ(c,λ) = ½g⁻¹∂g/∂c');
}

function drawHamiltonianContours(group, width, height) {
    // Draw concentric energy contours
    const numContours = 5;
    const centerX = width / 2;
    const centerY = height / 2;
    
    for (let i = 1; i <= numContours; i++) {
        const radius = (i / numContours) * Math.min(width, height) / 2;
        
        group.append('ellipse')
            .attr('cx', centerX)
            .attr('cy', centerY)
            .attr('rx', radius)
            .attr('ry', radius * 0.7)
            .attr('fill', 'none')
            .attr('stroke', '#f59e0b')
            .attr('stroke-width', 1.5)
            .attr('opacity', 0.7);
    }
    
    // Add equation label
    group.append('text')
        .attr('x', width / 2)
        .attr('y', -10)
        .attr('text-anchor', 'middle')
        .attr('class', 'math-text')
        .text('H = ½g(c,λ)v²');
}

function drawMetricNetwork(group, width) {
    const layers = [
        { nodes: 2, label: '[c,λ]', color: '#93c5fd' },
        { nodes: 4, label: '128', color: '#60a5fa' },
        { nodes: 5, label: '256', color: '#4dd0e1' },
        { nodes: 1, label: 'g(c,λ)', color: '#fbbf24' }
    ];
    
    const layerSpacing = width / (layers.length - 1);
    const nodeRadius = 12;
    
    layers.forEach((layer, layerIdx) => {
        const x = layerIdx * layerSpacing;
        const nodeSpacing = 200 / (layer.nodes + 1);
        
        for (let i = 0; i < layer.nodes; i++) {
            const y = (i + 1) * nodeSpacing;
            
            // Draw node
            group.append('circle')
                .attr('cx', x)
                .attr('cy', y)
                .attr('r', nodeRadius)
                .attr('fill', layer.color)
                .attr('stroke', '#374151')
                .attr('stroke-width', 1);
            
            // Connect to next layer
            if (layerIdx < layers.length - 1) {
                const nextLayer = layers[layerIdx + 1];
                const nextX = (layerIdx + 1) * layerSpacing;
                const nextNodeSpacing = 200 / (nextLayer.nodes + 1);
                
                for (let j = 0; j < nextLayer.nodes; j++) {
                    const nextY = (j + 1) * nextNodeSpacing;
                    
                    group.append('line')
                        .attr('x1', x + nodeRadius)
                        .attr('y1', y)
                        .attr('x2', nextX - nodeRadius)
                        .attr('y2', nextY)
                        .attr('class', 'network-link');
                }
            }
        }
        
        // Add layer label
        group.append('text')
            .attr('x', x)
            .attr('y', 230)
            .attr('text-anchor', 'middle')
            .attr('class', 'label-text')
            .text(layer.label);
    });
    
    // Add activation labels
    group.append('text')
        .attr('x', layerSpacing * 1.5)
        .attr('y', 250)
        .attr('text-anchor', 'middle')
        .attr('class', 'label-text')
        .attr('font-size', '10px')
        .text('tanh → softplus');
}

function drawFlowNetwork(group, width) {
    const layers = [
        { nodes: 3, label: '[c,v,λ]', color: '#86efac' },
        { nodes: 3, label: '64', color: '#4ade80' },
        { nodes: 4, label: '128', color: '#3b82f6' },
        { nodes: 1, label: 'Ȧ', color: '#fb923c' }
    ];
    
    const layerSpacing = width / (layers.length - 1);
    const nodeRadius = 12;
    
    layers.forEach((layer, layerIdx) => {
        const x = layerIdx * layerSpacing;
        const nodeSpacing = 200 / (layer.nodes + 1);
        
        for (let i = 0; i < layer.nodes; i++) {
            const y = (i + 1) * nodeSpacing;
            
            // Draw node
            group.append('circle')
                .attr('cx', x)
                .attr('cy', y)
                .attr('r', nodeRadius)
                .attr('fill', layer.color)
                .attr('stroke', '#374151')
                .attr('stroke-width', 1);
            
            // Connect to next layer
            if (layerIdx < layers.length - 1) {
                const nextLayer = layers[layerIdx + 1];
                const nextX = (layerIdx + 1) * layerSpacing;
                const nextNodeSpacing = 200 / (nextLayer.nodes + 1);
                
                for (let j = 0; j < nextLayer.nodes; j++) {
                    const nextY = (j + 1) * nextNodeSpacing;
                    
                    group.append('line')
                        .attr('x1', x + nodeRadius)
                        .attr('y1', y)
                        .attr('x2', nextX - nodeRadius)
                        .attr('y2', nextY)
                        .attr('class', 'network-link');
                }
            }
        }
        
        // Add layer label
        group.append('text')
            .attr('x', x)
            .attr('y', 230)
            .attr('text-anchor', 'middle')
            .attr('class', 'label-text')
            .text(layer.label);
    });
}

function drawMemoryComparison(group, width, height) {
    // Title
    group.append('text')
        .attr('x', width / 2)
        .attr('y', 20)
        .attr('text-anchor', 'middle')
        .attr('class', 'label-text')
        .attr('font-weight', 'bold')
        .text('Memory: Adjoint Method');
    
    // Traditional approach (left)
    const traditionalGroup = group.append('g')
        .attr('transform', 'translate(20, 40)');
    
    // Stack of memory blocks
    for (let i = 0; i < 5; i++) {
        traditionalGroup.append('rect')
            .attr('x', 0)
            .attr('y', i * 30)
            .attr('width', 80)
            .attr('height', 25)
            .attr('fill', '#fee2e2')
            .attr('stroke', '#dc2626')
            .attr('stroke-width', 1);
    }
    
    traditionalGroup.append('text')
        .attr('x', 40)
        .attr('y', 180)
        .attr('text-anchor', 'middle')
        .attr('class', 'math-text')
        .text('O(BT)');
    
    // Arrow
    group.append('path')
        .attr('d', 'M 120 100 L 160 100')
        .attr('stroke', '#374151')
        .attr('stroke-width', 2)
        .attr('marker-end', 'url(#arrowhead)');
    
    // Adjoint approach (right)
    const adjointGroup = group.append('g')
        .attr('transform', 'translate(180, 40)');
    
    // Single memory block
    adjointGroup.append('rect')
        .attr('x', 0)
        .attr('y', 50)
        .attr('width', 80)
        .attr('height', 25)
        .attr('fill', '#dcfce7')
        .attr('stroke', '#059669')
        .attr('stroke-width', 1);
    
    adjointGroup.append('text')
        .attr('x', 40)
        .attr('y', 180)
        .attr('text-anchor', 'middle')
        .attr('class', 'math-text')
        .text('O(B)');
    
    // Add checkmark
    group.append('text')
        .attr('x', width - 20)
        .attr('y', 100)
        .attr('font-size', '24px')
        .attr('fill', '#059669')
        .text('✓');
}

function drawTimeComparison(group, width, height) {
    // Title
    group.append('text')
        .attr('x', width / 2)
        .attr('y', 20)
        .attr('text-anchor', 'middle')
        .attr('class', 'label-text')
        .attr('font-weight', 'bold')
        .text('Time: Christoffel Caching');
    
    // Without cache (top)
    group.append('text')
        .attr('x', 20)
        .attr('y', 60)
        .attr('class', 'label-text')
        .text('Without cache:');
    
    group.append('text')
        .attr('x', 120)
        .attr('y', 60)
        .attr('class', 'math-text')
        .attr('fill', '#dc2626')
        .text('O(B·T·C_NN)');
    
    // With cache (bottom)
    group.append('text')
        .attr('x', 20)
        .attr('y', 100)
        .attr('class', 'label-text')
        .text('With cache:');
    
    group.append('text')
        .attr('x', 120)
        .attr('y', 100)
        .attr('class', 'math-text')
        .attr('fill', '#059669')
        .text('O(B·T)');
    
    // Grid visualization
    const gridGroup = group.append('g')
        .attr('transform', 'translate(20, 130)');
    
    const gridSize = 10;
    const cellSize = 20;
    
    // Color scale for cached values
    const colorScale = d3.scaleSequential()
        .domain([0, 1])
        .interpolator(d3.interpolateBlues);
    
    for (let i = 0; i < gridSize; i++) {
        for (let j = 0; j < gridSize; j++) {
            gridGroup.append('rect')
                .attr('x', i * cellSize)
                .attr('y', j * cellSize)
                .attr('width', cellSize - 1)
                .attr('height', cellSize - 1)
                .attr('fill', colorScale(Math.random()));
        }
    }
    
    // Interpolation arrows
    gridGroup.append('text')
        .attr('x', gridSize * cellSize + 10)
        .attr('y', gridSize * cellSize / 2)
        .attr('class', 'label-text')
        .attr('font-size', '10px')
        .text('Bilinear');
    
    gridGroup.append('text')
        .attr('x', gridSize * cellSize + 10)
        .attr('y', gridSize * cellSize / 2 + 15)
        .attr('class', 'label-text')
        .attr('font-size', '10px')
        .text('Interpolation');
}

function drawShootingAlgorithm(group, width, height) {
    // Draw artillery cannon
    const cannonGroup = group.append('g')
        .attr('transform', 'translate(30, 150)');
    
    // Cannon barrel
    cannonGroup.append('rect')
        .attr('x', 0)
        .attr('y', -10)
        .attr('width', 60)
        .attr('height', 20)
        .attr('fill', '#a3734f')
        .attr('stroke', '#8b6239')
        .attr('stroke-width', 1)
        .attr('transform', 'rotate(-30, 0, 0)');
    
    // Cannon wheel
    cannonGroup.append('circle')
        .attr('cx', 10)
        .attr('cy', 20)
        .attr('r', 15)
        .attr('fill', '#8b6239')
        .attr('stroke', '#6b4e2a')
        .attr('stroke-width', 2);
    
    // Multiple trajectory attempts
    const trajectories = [
        { angle: -45, opacity: 0.3 },
        { angle: -35, opacity: 0.5 },
        { angle: -30, opacity: 1.0 }, // Optimal
        { angle: -25, opacity: 0.5 },
        { angle: -15, opacity: 0.3 }
    ];
    
    trajectories.forEach(traj => {
        const path = d3.path();
        path.moveTo(50, 0);
        
        // Parabolic trajectory
        for (let t = 0; t <= 1; t += 0.05) {
            const x = 50 + t * 150;
            const y = Math.tan(traj.angle * Math.PI / 180) * (x - 50) + 
                      0.5 * 9.8 * Math.pow((x - 50) / 100, 2);
            path.lineTo(x, y);
        }
        
        cannonGroup.append('path')
            .attr('d', path.toString())
            .attr('fill', 'none')
            .attr('stroke', '#f97316')
            .attr('stroke-width', traj.opacity === 1 ? 2 : 1)
            .attr('opacity', traj.opacity);
    });
    
    // Target
    cannonGroup.append('circle')
        .attr('cx', 200)
        .attr('cy', 0)
        .attr('r', 8)
        .attr('fill', '#dc2626')
        .attr('stroke', '#991b1b')
        .attr('stroke-width', 2);
    
    // Label
    group.append('text')
        .attr('x', width / 2)
        .attr('y', 20)
        .attr('text-anchor', 'middle')
        .attr('class', 'label-text')
        .text('Boundary Value Problem');
}

function drawSpectralReconstruction(group, width, height, data) {
    if (!data) return;
    
    // Create scales
    const xScale = d3.scaleLinear()
        .domain([200, 800])
        .range([0, width]);
    
    const yScale = d3.scaleLinear()
        .domain([0, 0.3])
        .range([height, 0]);
    
    // Draw all 6 concentration spectra
    const colors = ['#dc2626', '#f97316', '#eab308', '#059669', '#3b82f6', '#8b5cf6'];
    
    data.concentrations.forEach((conc, idx) => {
        const spectrumData = data.getSpectrumPoints(conc);
        if (!spectrumData) return;
        
        // Sample every 20th point for performance
        const sampledData = spectrumData.filter((d, i) => i % 20 === 0);
        
        const line = d3.line()
            .x(d => xScale(d.wavelength))
            .y(d => yScale(d.absorbance))
            .curve(d3.curveBasis);
        
        group.append('path')
            .datum(sampledData)
            .attr('d', line)
            .attr('fill', 'none')
            .attr('stroke', colors[idx])
            .attr('stroke-width', 1.5)
            .attr('opacity', 0.8);
    });
    
    // Add axis
    group.append('line')
        .attr('x1', 0)
        .attr('y1', height)
        .attr('x2', width)
        .attr('y2', height)
        .attr('stroke', '#6b7280')
        .attr('stroke-width', 1);
    
    // Labels
    group.append('text')
        .attr('x', width / 2)
        .attr('y', height + 20)
        .attr('text-anchor', 'middle')
        .attr('class', 'label-text')
        .attr('font-size', '10px')
        .text('Wavelength (nm)');
}

function drawColorMapping(group, width, height) {
    // Simple color transformation visualization
    const steps = [
        { x: 0, label: 'Spectral', color: '#8b5cf6' },
        { x: width / 3, label: 'CIE XYZ', color: '#3b82f6' },
        { x: 2 * width / 3, label: 'RGB', color: '#059669' }
    ];
    
    steps.forEach((step, idx) => {
        // Draw box
        group.append('rect')
            .attr('x', step.x)
            .attr('y', 20)
            .attr('width', 60)
            .attr('height', 40)
            .attr('fill', step.color)
            .attr('opacity', 0.2)
            .attr('stroke', step.color)
            .attr('stroke-width', 2);
        
        // Label
        group.append('text')
            .attr('x', step.x + 30)
            .attr('y', 75)
            .attr('text-anchor', 'middle')
            .attr('class', 'label-text')
            .attr('font-size', '10px')
            .text(step.label);
        
        // Arrow to next
        if (idx < steps.length - 1) {
            group.append('path')
                .attr('d', `M ${step.x + 65} 40 L ${steps[idx + 1].x - 5} 40`)
                .attr('stroke', '#6b7280')
                .attr('stroke-width', 1)
                .attr('marker-end', 'url(#arrowhead)');
        }
    });
}