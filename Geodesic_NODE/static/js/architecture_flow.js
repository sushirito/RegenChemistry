/**
 * Architecture Flow - D3.js Pipeline Visualization
 * Shows the data flow through the geodesic computation pipeline
 */

let pipelineSvg, pipelineData;
let flowAnimation = null;

function initArchitectureFlow() {
    const container = d3.select('#pipeline-svg');
    const width = container.node().getBoundingClientRect().width;
    const height = container.node().getBoundingClientRect().height;
    
    pipelineSvg = container
        .attr('width', width)
        .attr('height', height);
    
    // Create gradient definitions
    createGradients();
    
    // Load and render architecture
    loadArchitectureData();
}

function createGradients() {
    const defs = pipelineSvg.append('defs');
    
    // Neural network gradient
    const neuralGradient = defs.append('linearGradient')
        .attr('id', 'neural-gradient')
        .attr('x1', '0%')
        .attr('y1', '0%')
        .attr('x2', '100%')
        .attr('y2', '0%');
    
    neuralGradient.append('stop')
        .attr('offset', '0%')
        .style('stop-color', '#00d4ff')
        .style('stop-opacity', 1);
    
    neuralGradient.append('stop')
        .attr('offset', '100%')
        .style('stop-color', '#ff00ff')
        .style('stop-opacity', 1);
    
    // Flow gradient
    const flowGradient = defs.append('linearGradient')
        .attr('id', 'flow-gradient')
        .attr('x1', '0%')
        .attr('y1', '0%')
        .attr('x2', '100%')
        .attr('y2', '0%');
    
    flowGradient.append('stop')
        .attr('offset', '0%')
        .style('stop-color', '#ffd700')
        .style('stop-opacity', 0.8);
    
    flowGradient.append('stop')
        .attr('offset', '100%')
        .style('stop-color', '#ff6b6b')
        .style('stop-opacity', 0.8);
    
    // Glow filter
    const filter = defs.append('filter')
        .attr('id', 'glow');
    
    filter.append('feGaussianBlur')
        .attr('stdDeviation', '3')
        .attr('result', 'coloredBlur');
    
    const feMerge = filter.append('feMerge');
    feMerge.append('feMergeNode')
        .attr('in', 'coloredBlur');
    feMerge.append('feMergeNode')
        .attr('in', 'SourceGraphic');
}

async function loadArchitectureData() {
    try {
        const response = await fetch('/api/network_architecture');
        pipelineData = await response.json();
        renderPipeline();
        startFlowAnimation();
    } catch (error) {
        console.error('Error loading architecture:', error);
        renderDefaultPipeline();
    }
}

function renderPipeline() {
    const width = pipelineSvg.node().getBoundingClientRect().width;
    const height = pipelineSvg.node().getBoundingClientRect().height;
    
    // Clear existing content
    pipelineSvg.selectAll('g.pipeline-group').remove();
    
    const g = pipelineSvg.append('g')
        .attr('class', 'pipeline-group');
    
    // Pipeline steps
    const steps = pipelineData.pipeline;
    const stepWidth = width / (steps.length + 1);
    const stepY = height / 2;
    
    // Draw connections first (behind nodes)
    const connections = g.append('g').attr('class', 'connections');
    
    steps.forEach((step, i) => {
        if (i < steps.length - 1) {
            // Curved path between steps
            const x1 = (i + 1) * stepWidth;
            const x2 = (i + 2) * stepWidth;
            
            const path = d3.path();
            path.moveTo(x1, stepY);
            path.quadraticCurveTo(
                (x1 + x2) / 2, stepY - 30,
                x2, stepY
            );
            
            connections.append('path')
                .attr('d', path.toString())
                .attr('stroke', 'url(#flow-gradient)')
                .attr('stroke-width', 2)
                .attr('fill', 'none')
                .attr('opacity', 0.6)
                .attr('class', `connection-${i}`);
            
            // Animated dots along path
            connections.append('circle')
                .attr('r', 4)
                .attr('fill', '#ffd700')
                .attr('filter', 'url(#glow)')
                .attr('class', `flow-dot-${i}`)
                .append('animateMotion')
                .attr('dur', '3s')
                .attr('repeatCount', 'indefinite')
                .attr('path', path.toString());
        }
    });
    
    // Draw pipeline nodes
    const nodes = g.append('g').attr('class', 'nodes');
    
    steps.forEach((step, i) => {
        const x = (i + 1) * stepWidth;
        
        const nodeGroup = nodes.append('g')
            .attr('class', `node-${i}`)
            .attr('transform', `translate(${x}, ${stepY})`);
        
        // Node circle
        nodeGroup.append('circle')
            .attr('r', 25)
            .attr('fill', 'rgba(20, 30, 48, 0.9)')
            .attr('stroke', 'url(#neural-gradient)')
            .attr('stroke-width', 2)
            .attr('filter', 'url(#glow)');
        
        // Node icon/symbol
        nodeGroup.append('text')
            .attr('text-anchor', 'middle')
            .attr('dy', 5)
            .attr('fill', '#00d4ff')
            .attr('font-size', '20px')
            .text(getStepIcon(step.step));
        
        // Step label
        nodeGroup.append('text')
            .attr('text-anchor', 'middle')
            .attr('y', -40)
            .attr('fill', '#fff')
            .attr('font-size', '12px')
            .attr('font-weight', 'bold')
            .text(step.step);
        
        // Mathematical notation
        nodeGroup.append('foreignObject')
            .attr('x', -60)
            .attr('y', 35)
            .attr('width', 120)
            .attr('height', 40)
            .append('xhtml:div')
            .style('text-align', 'center')
            .style('color', 'rgba(255, 255, 255, 0.7)')
            .style('font-size', '10px')
            .style('font-family', 'monospace')
            .html(step.description);
        
        // Hover interaction
        nodeGroup
            .on('mouseenter', function() {
                d3.select(this).select('circle')
                    .transition()
                    .duration(200)
                    .attr('r', 30)
                    .attr('stroke-width', 3);
                
                // Show detailed math
                showPipelineMath(step);
            })
            .on('mouseleave', function() {
                d3.select(this).select('circle')
                    .transition()
                    .duration(200)
                    .attr('r', 25)
                    .attr('stroke-width', 2);
            });
    });
    
    // Add network architecture details
    renderNetworkDetails();
}

function getStepIcon(stepName) {
    const icons = {
        'Input': '📥',
        'Metric Learning': '🔍',
        'Christoffel': 'Γ',
        'Geodesic ODE': '∫',
        'Shooting Method': '🎯',
        'Path Features': '📊',
        'Decoder': '🧠'
    };
    return icons[stepName] || '•';
}

function renderNetworkDetails() {
    const width = pipelineSvg.node().getBoundingClientRect().width;
    const height = pipelineSvg.node().getBoundingClientRect().height;
    
    // Metric Network visualization
    const metricNet = pipelineSvg.append('g')
        .attr('class', 'metric-network')
        .attr('transform', `translate(${width * 0.2}, ${height * 0.15})`);
    
    drawNeuralNetwork(metricNet, pipelineData.metric_network, 'Metric Network g(c,λ)');
    
    // Decoder Network visualization
    const decoderNet = pipelineSvg.append('g')
        .attr('class', 'decoder-network')
        .attr('transform', `translate(${width * 0.7}, ${height * 0.15})`);
    
    drawNeuralNetwork(decoderNet, pipelineData.decoder_network, 'Absorbance Decoder');
}

function drawNeuralNetwork(container, networkData, title) {
    const layers = networkData.layers;
    const layerSpacing = 40;
    const nodeRadius = 4;
    
    // Title
    container.append('text')
        .attr('x', 0)
        .attr('y', -10)
        .attr('text-anchor', 'middle')
        .attr('fill', '#00d4ff')
        .attr('font-size', '11px')
        .attr('font-weight', 'bold')
        .text(title);
    
    // Draw layers
    layers.forEach((layer, i) => {
        const x = i * layerSpacing - (layers.length - 1) * layerSpacing / 2;
        const nodesInLayer = Math.min(layer.size, 8); // Limit visual nodes
        
        for (let j = 0; j < nodesInLayer; j++) {
            const y = (j - nodesInLayer / 2) * 10 + 20;
            
            // Node
            container.append('circle')
                .attr('cx', x)
                .attr('cy', y)
                .attr('r', nodeRadius)
                .attr('fill', 'url(#neural-gradient)')
                .attr('opacity', 0.8);
            
            // Connections to next layer
            if (i < layers.length - 1) {
                const nextLayer = layers[i + 1];
                const nextNodesInLayer = Math.min(nextLayer.size, 8);
                const nextX = (i + 1) * layerSpacing - (layers.length - 1) * layerSpacing / 2;
                
                for (let k = 0; k < nextNodesInLayer; k++) {
                    const nextY = (k - nextNodesInLayer / 2) * 10 + 20;
                    
                    container.append('line')
                        .attr('x1', x)
                        .attr('y1', y)
                        .attr('x2', nextX)
                        .attr('y2', nextY)
                        .attr('stroke', 'rgba(0, 212, 255, 0.2)')
                        .attr('stroke-width', 0.5);
                }
            }
        }
        
        // Layer label
        container.append('text')
            .attr('x', x)
            .attr('y', 45)
            .attr('text-anchor', 'middle')
            .attr('fill', 'rgba(255, 255, 255, 0.6)')
            .attr('font-size', '9px')
            .text(`${layer.name} (${layer.size})`);
        
        // Activation function
        if (layer.activation) {
            container.append('text')
                .attr('x', x)
                .attr('y', 55)
                .attr('text-anchor', 'middle')
                .attr('fill', 'rgba(255, 212, 0, 0.6)')
                .attr('font-size', '8px')
                .text(layer.activation);
        }
    });
    
    // Parameter count
    container.append('text')
        .attr('x', 0)
        .attr('y', 70)
        .attr('text-anchor', 'middle')
        .attr('fill', 'rgba(255, 255, 255, 0.5)')
        .attr('font-size', '9px')
        .text(`Parameters: ${networkData.parameters.toLocaleString()}`);
}

function startFlowAnimation() {
    let step = 0;
    const steps = pipelineData.pipeline.length;
    
    flowAnimation = setInterval(() => {
        // Highlight current step
        pipelineSvg.selectAll('.nodes g')
            .transition()
            .duration(300)
            .attr('opacity', (d, i) => i === step ? 1 : 0.5);
        
        // Pulse effect on current node
        pipelineSvg.select(`.node-${step} circle`)
            .transition()
            .duration(300)
            .attr('r', 30)
            .transition()
            .duration(300)
            .attr('r', 25);
        
        // Highlight connection
        if (step < steps - 1) {
            pipelineSvg.select(`.connection-${step}`)
                .transition()
                .duration(300)
                .attr('stroke-width', 4)
                .attr('opacity', 1)
                .transition()
                .duration(300)
                .attr('stroke-width', 2)
                .attr('opacity', 0.6);
        }
        
        step = (step + 1) % steps;
    }, 2000);
}

function showPipelineMath(step) {
    const mathOverlay = document.getElementById('pipeline-math');
    const equationDiv = document.getElementById('pipeline-equation');
    
    // Define mathematical equations for each step
    const equations = {
        'Input': '[c_0, c_1, \\lambda] \\in \\mathbb{R}^3',
        'Metric Learning': 'g(c,\\lambda) = \\text{softplus}(\\mathcal{N}_\\theta(c,\\lambda)) + 0.1',
        'Christoffel': '\\Gamma = \\frac{1}{2g} \\frac{\\partial g}{\\partial c}',
        'Geodesic ODE': '\\frac{d^2c}{dt^2} + \\Gamma \\left(\\frac{dc}{dt}\\right)^2 = 0',
        'Shooting Method': '\\min_{v_0} \\|\\gamma(1; c_0, v_0) - c_1\\|^2',
        'Path Features': '\\mathbf{f} = [c_f, \\bar{c}, L, v_{\\max}, \\lambda]',
        'Decoder': 'A = \\mathcal{D}_\\phi(\\mathbf{f})'
    };
    
    const equation = equations[step.step] || '';
    if (equation) {
        katex.render(equation, equationDiv, {
            throwOnError: false,
            displayMode: true
        });
        
        mathOverlay.classList.add('show');
        
        // Hide after 3 seconds
        setTimeout(() => {
            mathOverlay.classList.remove('show');
        }, 3000);
    }
}

function updateFeatureExtraction(geodesicData) {
    if (!geodesicData || !geodesicData.path) return;
    
    // Calculate path features
    const path = geodesicData.path;
    const concentrations = path.map(p => p[0]);
    const velocities = path.map(p => p[1]);
    
    const features = {
        c_final: concentrations[concentrations.length - 1],
        c_mean: concentrations.reduce((a, b) => a + b) / concentrations.length,
        path_length: geodesicData.path_length,
        v_max: Math.max(...velocities.map(Math.abs)),
        wavelength: parseFloat(document.getElementById('wavelength').value)
    };
    
    // Visualize feature extraction
    visualizeFeatures(features);
}

function visualizeFeatures(features) {
    const container = d3.select('#features-svg');
    container.selectAll('*').remove();
    
    const width = container.node().getBoundingClientRect().width;
    const height = container.node().getBoundingClientRect().height;
    
    const g = container.append('g')
        .attr('transform', `translate(${width/2}, ${height/2})`);
    
    // Feature names and values
    const featureData = [
        { name: 'c_final', value: features.c_final.toFixed(2), unit: 'ppb' },
        { name: 'c_mean', value: features.c_mean.toFixed(2), unit: 'ppb' },
        { name: 'L_path', value: features.path_length.toFixed(3), unit: '' },
        { name: 'v_max', value: features.v_max.toFixed(3), unit: 'ppb/t' },
        { name: 'λ', value: features.wavelength.toFixed(0), unit: 'nm' }
    ];
    
    // Radial layout for features
    const angleStep = (2 * Math.PI) / featureData.length;
    const radius = Math.min(width, height) * 0.3;
    
    featureData.forEach((feature, i) => {
        const angle = i * angleStep - Math.PI / 2;
        const x = Math.cos(angle) * radius;
        const y = Math.sin(angle) * radius;
        
        const featureGroup = g.append('g')
            .attr('transform', `translate(${x}, ${y})`);
        
        // Feature circle
        featureGroup.append('circle')
            .attr('r', 30)
            .attr('fill', 'rgba(20, 30, 48, 0.9)')
            .attr('stroke', '#ffd700')
            .attr('stroke-width', 2);
        
        // Feature value
        featureGroup.append('text')
            .attr('text-anchor', 'middle')
            .attr('dy', 0)
            .attr('fill', '#ffd700')
            .attr('font-size', '14px')
            .attr('font-weight', 'bold')
            .text(feature.value);
        
        // Feature name
        featureGroup.append('text')
            .attr('text-anchor', 'middle')
            .attr('y', -45)
            .attr('fill', '#fff')
            .attr('font-size', '11px')
            .text(feature.name);
        
        // Unit
        featureGroup.append('text')
            .attr('text-anchor', 'middle')
            .attr('y', 15)
            .attr('fill', 'rgba(255, 255, 255, 0.6)')
            .attr('font-size', '9px')
            .text(feature.unit);
        
        // Connection to center
        g.append('line')
            .attr('x1', 0)
            .attr('y1', 0)
            .attr('x2', x * 0.6)
            .attr('y2', y * 0.6)
            .attr('stroke', 'rgba(255, 215, 0, 0.3)')
            .attr('stroke-width', 1);
    });
    
    // Center node
    g.append('circle')
        .attr('r', 20)
        .attr('fill', 'rgba(0, 212, 255, 0.2)')
        .attr('stroke', '#00d4ff')
        .attr('stroke-width', 2);
    
    g.append('text')
        .attr('text-anchor', 'middle')
        .attr('dy', 5)
        .attr('fill', '#00d4ff')
        .attr('font-size', '16px')
        .text('𝒇');
}

function renderDefaultPipeline() {
    // Fallback visualization if data loading fails
    const width = pipelineSvg.node().getBoundingClientRect().width;
    const height = pipelineSvg.node().getBoundingClientRect().height;
    
    pipelineSvg.append('text')
        .attr('x', width / 2)
        .attr('y', height / 2)
        .attr('text-anchor', 'middle')
        .attr('fill', 'rgba(255, 255, 255, 0.5)')
        .text('Loading architecture...');
}

// Export functions
window.updateFeatureExtraction = updateFeatureExtraction;