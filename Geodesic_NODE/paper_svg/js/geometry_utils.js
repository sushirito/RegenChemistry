// Geometry utilities for geodesic visualizations

// Create a curved manifold surface using parametric equations
function createManifoldSurface(width, height) {
    const points = [];
    const rows = 20;
    const cols = 30;
    
    for (let i = 0; i <= rows; i++) {
        const row = [];
        for (let j = 0; j <= cols; j++) {
            const u = j / cols;
            const v = i / rows;
            
            // Create a curved surface with some warping
            const x = u * width;
            const y = v * height;
            const z = Math.sin(u * Math.PI * 2) * Math.cos(v * Math.PI) * 30;
            
            row.push([x, y, z]);
        }
        points.push(row);
    }
    
    return points;
}

// Generate geodesic path on manifold
function generateGeodesicPath(start, end, numPoints = 50) {
    const path = [];
    
    for (let i = 0; i <= numPoints; i++) {
        const t = i / numPoints;
        
        // Geodesic follows the surface curvature
        const x = start[0] + (end[0] - start[0]) * t;
        const y = start[1] + (end[1] - start[1]) * t;
        
        // Add curvature based on the manifold geometry
        const curvature = Math.sin(t * Math.PI) * 20;
        const lateralShift = Math.sin(t * Math.PI * 2) * 10;
        
        path.push([
            x + lateralShift,
            y + curvature,
            Math.sin(x / 100) * Math.cos(y / 100) * 15
        ]);
    }
    
    return path;
}

// Generate Euclidean (straight) path
function generateEuclideanPath(start, end, numPoints = 50) {
    const path = [];
    
    for (let i = 0; i <= numPoints; i++) {
        const t = i / numPoints;
        
        // Straight line interpolation
        const x = start[0] + (end[0] - start[0]) * t;
        const y = start[1] + (end[1] - start[1]) * t;
        const z = start[2] + (end[2] - start[2]) * t;
        
        path.push([x, y, z]);
    }
    
    return path;
}

// Convert 3D coordinates to 2D isometric projection
function toIsometric(point3D) {
    const [x, y, z] = point3D;
    const angle = Math.PI / 6; // 30 degrees
    
    const isoX = (x - y) * Math.cos(angle);
    const isoY = (x + y) * Math.sin(angle) - z;
    
    return [isoX, isoY];
}

// Generate great circle path on sphere
function generateGreatCirclePath(startLat, startLon, endLat, endLon, numPoints = 50) {
    const path = [];
    
    // Convert to radians
    const lat1 = startLat * Math.PI / 180;
    const lon1 = startLon * Math.PI / 180;
    const lat2 = endLat * Math.PI / 180;
    const lon2 = endLon * Math.PI / 180;
    
    for (let i = 0; i <= numPoints; i++) {
        const f = i / numPoints;
        
        // Spherical interpolation
        const A = Math.sin((1 - f) * Math.acos(
            Math.sin(lat1) * Math.sin(lat2) + 
            Math.cos(lat1) * Math.cos(lat2) * Math.cos(lon2 - lon1)
        )) / Math.sin(Math.acos(
            Math.sin(lat1) * Math.sin(lat2) + 
            Math.cos(lat1) * Math.cos(lat2) * Math.cos(lon2 - lon1)
        ));
        
        const B = Math.sin(f * Math.acos(
            Math.sin(lat1) * Math.sin(lat2) + 
            Math.cos(lat1) * Math.cos(lat2) * Math.cos(lon2 - lon1)
        )) / Math.sin(Math.acos(
            Math.sin(lat1) * Math.sin(lat2) + 
            Math.cos(lat1) * Math.cos(lat2) * Math.cos(lon2 - lon1)
        ));
        
        const x = A * Math.cos(lat1) * Math.cos(lon1) + B * Math.cos(lat2) * Math.cos(lon2);
        const y = A * Math.cos(lat1) * Math.sin(lon1) + B * Math.cos(lat2) * Math.sin(lon2);
        const z = A * Math.sin(lat1) + B * Math.sin(lat2);
        
        // Convert to latitude/longitude
        const lat = Math.atan2(z, Math.sqrt(x * x + y * y)) * 180 / Math.PI;
        const lon = Math.atan2(y, x) * 180 / Math.PI;
        
        path.push([lon, lat]);
    }
    
    return path;
}

// Generate Christoffel symbol vector field
function generateChristoffelField(width, height, gridSize = 20) {
    const vectors = [];
    
    for (let i = 0; i < gridSize; i++) {
        for (let j = 0; j < gridSize; j++) {
            const x = (i / gridSize) * width;
            const y = (j / gridSize) * height;
            
            // Simulate Christoffel symbols based on metric tensor derivatives
            const magnitude = Math.sin(x / width * Math.PI) * Math.cos(y / height * Math.PI);
            const angle = Math.atan2(y - height/2, x - width/2) + Math.PI/4;
            
            vectors.push({
                x: x,
                y: y,
                dx: Math.cos(angle) * magnitude * 15,
                dy: Math.sin(angle) * magnitude * 15,
                magnitude: Math.abs(magnitude)
            });
        }
    }
    
    return vectors;
}

// Generate metric tensor heatmap data
function generateMetricTensorData(width, height, resolution = 50) {
    const data = [];
    
    for (let i = 0; i < resolution; i++) {
        for (let j = 0; j < resolution; j++) {
            const c = i / resolution; // Normalized concentration
            const lambda = j / resolution; // Normalized wavelength
            
            // Simulate metric tensor value based on spectral sensitivity
            // Higher values where spectral response changes rapidly
            const value = Math.exp(-Math.pow(c - 0.5, 2) * 4) * 
                         Math.sin(lambda * Math.PI * 3) +
                         Math.exp(-Math.pow(lambda - 0.3, 2) * 10) * 
                         (1 - Math.abs(c - 0.5));
            
            data.push({
                x: i,
                y: j,
                c: c * 60, // Scale to actual concentration range
                lambda: 200 + lambda * 600, // Scale to wavelength range
                value: value
            });
        }
    }
    
    return data;
}