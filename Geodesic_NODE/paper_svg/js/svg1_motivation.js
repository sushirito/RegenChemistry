// SVG 1: Problem Motivation - Why Interpolation Requires Geometry
function createSVG1(data) {
    const width = 800;
    const height = 500;
    const margin = { top: 40, right: 40, bottom: 40, left: 40 };
    
    // Create main SVG
    const svg = d3.select('#svg1-container')
        .append('svg')
        .attr('width', width)
        .attr('height', height)
        .attr('viewBox', `0 0 ${width} ${height}`);
    
    // Define gradients
    const defs = svg.append('defs');
    
    // Manifold gradient
    const manifoldGradient = defs.append('linearGradient')
        .attr('id', 'manifoldGradient')
        .attr('x1', '0%').attr('y1', '0%')
        .attr('x2', '100%').attr('y2', '100%');
    
    manifoldGradient.append('stop')
        .attr('offset', '0%')
        .attr('style', 'stop-color:#1e3a8a;stop-opacity:0.8');
    
    manifoldGradient.append('stop')
        .attr('offset', '100%')
        .attr('style', 'stop-color:#14b8a6;stop-opacity:0.8');
    
    // Main manifold visualization (left 60%)
    const manifoldGroup = svg.append('g')
        .attr('transform', `translate(${margin.left}, ${margin.top})`);
    
    const manifoldWidth = width * 0.6;
    const manifoldHeight = height - margin.top - margin.bottom;
    
    // Create manifold surface
    drawManifoldSurface(manifoldGroup, manifoldWidth, manifoldHeight);
    
    // Add concentration-wavelength space label
    manifoldGroup.append('text')
        .attr('x', 10)
        .attr('y', 20)
        .attr('class', 'header-text')
        .text('CWA Space');
    
    // Draw paths
    const startPoint = [100, 200];
    const endPoint = [400, 250];
    
    // Euclidean path (straight line)
    drawEuclideanPath(manifoldGroup, startPoint, endPoint);
    
    // Geodesic path (curved)
    drawGeodesicPath(manifoldGroup, startPoint, endPoint);
    
    // Add path labels
    manifoldGroup.append('text')
        .attr('x', 250)
        .attr('y', 180)
        .attr('class', 'label-text')
        .attr('fill', '#dc2626')
        .text('Euclidean Path');
    
    manifoldGroup.append('text')
        .attr('x', 250)
        .attr('y', 280)
        .attr('class', 'label-text')
        .attr('fill', '#059669')
        .text('Geodesic Path');
    
    // Add anchor points with spectral data
    const anchorConcentrations = [0, 20, 40, 60];
    const anchorPositions = [
        [150, 150],
        [250, 200],
        [350, 250],
        [450, 300]
    ];
    
    anchorConcentrations.forEach((conc, i) => {
        const pos = anchorPositions[i];
        
        // Draw anchor point
        manifoldGroup.append('circle')
            .attr('cx', pos[0])
            .attr('cy', pos[1])
            .attr('r', 8)
            .attr('class', 'anchor-point');
        
        // Add concentration label
        manifoldGroup.append('text')
            .attr('x', pos[0])
            .attr('y', pos[1] - 15)
            .attr('class', 'math-text')
            .attr('text-anchor', 'middle')
            .text(`c${i+1} = ${conc} ppb`);
        
        // Draw mini spectrum
        drawMiniSpectrum(manifoldGroup, data, conc, pos[0] - 30, pos[1] - 80, 60, 40);
    });
    
    // Great Circle Inset (top-right)
    const earthGroup = svg.append('g')
        .attr('transform', `translate(${width - 200}, ${50})`);
    
    drawEarthGreatCircle(earthGroup);
    
    // Add title for great circle
    earthGroup.append('text')
        .attr('x', 75)
        .attr('y', -10)
        .attr('class', 'label-text')
        .attr('text-anchor', 'middle')
        .text('Same Principle');
}

function drawManifoldSurface(group, width, height) {
    // Create grid pattern for manifold
    const gridSize = 20;
    
    for (let i = 0; i <= gridSize; i++) {
        const x = (i / gridSize) * width;
        
        // Vertical grid lines with curvature
        const path = d3.path();
        path.moveTo(x, 0);
        
        for (let j = 0; j <= 20; j++) {
            const y = (j / 20) * height;
            const curve = Math.sin((x / width) * Math.PI) * Math.cos((y / height) * Math.PI) * 20;
            path.lineTo(x + curve, y);
        }
        
        group.append('path')
            .attr('d', path.toString())
            .attr('class', 'grid-line')
            .attr('stroke', 'rgba(255,255,255,0.2)');
    }
    
    for (let i = 0; i <= gridSize; i++) {
        const y = (i / gridSize) * height;
        
        // Horizontal grid lines with curvature
        const path = d3.path();
        path.moveTo(0, y);
        
        for (let j = 0; j <= 20; j++) {
            const x = (j / 20) * width;
            const curve = Math.sin((x / width) * Math.PI) * Math.cos((y / height) * Math.PI) * 20;
            path.lineTo(x, y + curve);
        }
        
        group.append('path')
            .attr('d', path.toString())
            .attr('class', 'grid-line')
            .attr('stroke', 'rgba(255,255,255,0.2)');
    }
    
    // Add gradient fill for depth
    group.append('rect')
        .attr('width', width)
        .attr('height', height)
        .attr('fill', 'url(#manifoldGradient)')
        .attr('opacity', 0.3);
}

function drawEuclideanPath(group, start, end) {
    const line = d3.line()
        .x(d => d[0])
        .y(d => d[1]);
    
    const pathData = [start, end];
    
    group.append('path')
        .attr('d', line(pathData))
        .attr('class', 'euclidean-path')
        .attr('stroke', '#dc2626')
        .attr('stroke-width', 3)
        .attr('stroke-dasharray', '5,3')
        .attr('fill', 'none');
    
    // Add arrow showing it "cuts through"
    group.append('text')
        .attr('x', (start[0] + end[0]) / 2 - 20)
        .attr('y', (start[1] + end[1]) / 2 - 10)
        .attr('font-size', '20px')
        .attr('fill', '#dc2626')
        .text('✂');
}

function drawGeodesicPath(group, start, end) {
    const path = d3.path();
    path.moveTo(start[0], start[1]);
    
    // Create curved geodesic path
    const controlPoint1 = [start[0] + 100, start[1] + 50];
    const controlPoint2 = [end[0] - 100, end[1] + 30];
    
    path.bezierCurveTo(
        controlPoint1[0], controlPoint1[1],
        controlPoint2[0], controlPoint2[1],
        end[0], end[1]
    );
    
    group.append('path')
        .attr('d', path.toString())
        .attr('class', 'geodesic-path')
        .attr('stroke', '#059669')
        .attr('stroke-width', 3)
        .attr('fill', 'none');
}

function drawMiniSpectrum(group, data, concentration, x, y, width, height) {
    if (!data) return;
    
    const spectrumData = data.getSpectrumPoints(concentration);
    if (!spectrumData) return;
    
    // Filter to show key features
    const filteredData = spectrumData.filter((d, i) => i % 10 === 0);
    
    // Create scales
    const xScale = d3.scaleLinear()
        .domain([200, 800])
        .range([0, width]);
    
    const yScale = d3.scaleLinear()
        .domain([0, 0.3])
        .range([height, 0]);
    
    // Create container
    const spectrumGroup = group.append('g')
        .attr('transform', `translate(${x}, ${y})`);
    
    // Add background
    spectrumGroup.append('rect')
        .attr('width', width)
        .attr('height', height)
        .attr('fill', 'white')
        .attr('stroke', '#e5e7eb')
        .attr('stroke-width', 1);
    
    // Draw spectrum line
    const line = d3.line()
        .x(d => xScale(d.wavelength))
        .y(d => yScale(d.absorbance))
        .curve(d3.curveMonotoneX);
    
    spectrumGroup.append('path')
        .datum(filteredData)
        .attr('d', line)
        .attr('fill', 'none')
        .attr('stroke', '#3b82f6')
        .attr('stroke-width', 1.5);
}

function drawEarthGreatCircle(group) {
    const radius = 70;
    
    // Draw Earth circle
    group.append('circle')
        .attr('cx', radius)
        .attr('cy', radius)
        .attr('r', radius)
        .attr('fill', '#0ea5e9')
        .attr('stroke', '#6b7280')
        .attr('stroke-width', 1);
    
    // Add simple continents
    const continents = group.append('g');
    
    // Simplified continent shapes
    continents.append('ellipse')
        .attr('cx', radius - 20)
        .attr('cy', radius - 10)
        .attr('rx', 25)
        .attr('ry', 20)
        .attr('fill', '#22c55e');
    
    continents.append('ellipse')
        .attr('cx', radius + 15)
        .attr('cy', radius + 10)
        .attr('rx', 20)
        .attr('ry', 15)
        .attr('fill', '#22c55e');
    
    // Draw great circle arc
    const arc = d3.arc()
        .innerRadius(radius - 1)
        .outerRadius(radius + 1)
        .startAngle(-Math.PI / 3)
        .endAngle(Math.PI / 3);
    
    group.append('path')
        .attr('d', arc)
        .attr('transform', `translate(${radius}, ${radius})`)
        .attr('fill', '#f97316');
    
    // Add grid lines
    for (let i = 1; i < 6; i++) {
        // Latitude lines
        const latRadius = radius * (i / 6);
        group.append('circle')
            .attr('cx', radius)
            .attr('cy', radius)
            .attr('r', latRadius)
            .attr('fill', 'none')
            .attr('stroke', '#e5e7eb')
            .attr('stroke-width', 0.5);
    }
    
    // Longitude lines
    for (let i = 0; i < 8; i++) {
        const angle = (i / 8) * Math.PI * 2;
        group.append('line')
            .attr('x1', radius)
            .attr('y1', radius)
            .attr('x2', radius + Math.cos(angle) * radius)
            .attr('y2', radius + Math.sin(angle) * radius)
            .attr('stroke', '#e5e7eb')
            .attr('stroke-width', 0.5);
    }
    
    // Add city labels
    group.append('text')
        .attr('x', radius - 30)
        .attr('y', radius)
        .attr('font-size', '10px')
        .attr('fill', '#374151')
        .text('SF');
    
    group.append('text')
        .attr('x', radius + 20)
        .attr('y', radius)
        .attr('font-size', '10px')
        .attr('fill', '#374151')
        .text('Tokyo');
}