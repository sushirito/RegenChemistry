/**
 * Manifold Renderer - Three.js 3D Visualization
 * Renders the Riemannian manifold with geodesic paths
 */

let manifoldScene, manifoldCamera, manifoldRenderer;
let manifoldMesh, geodesicLines = [];
let manifoldControls;
let manifoldData = null;
let animationFrame = null;

function initManifoldRenderer() {
    const container = document.getElementById('manifold-3d');
    const width = container.clientWidth;
    const height = container.clientHeight;
    
    // Scene setup
    manifoldScene = new THREE.Scene();
    manifoldScene.background = new THREE.Color(0x0a0e27);
    manifoldScene.fog = new THREE.Fog(0x0a0e27, 100, 500);
    
    // Camera setup
    manifoldCamera = new THREE.PerspectiveCamera(
        60,
        width / height,
        0.1,
        1000
    );
    manifoldCamera.position.set(100, 80, 100);
    manifoldCamera.lookAt(0, 0, 0);
    
    // Renderer setup
    manifoldRenderer = new THREE.WebGLRenderer({
        antialias: true,
        alpha: true
    });
    manifoldRenderer.setSize(width, height);
    manifoldRenderer.setPixelRatio(window.devicePixelRatio);
    container.appendChild(manifoldRenderer.domElement);
    
    // Lighting
    setupManifoldLighting();
    
    // Controls (orbit)
    setupManifoldControls();
    
    // Load and create manifold mesh
    loadManifoldMesh();
    
    // Start animation loop
    animateManifold();
    
    // Handle resize
    window.addEventListener('resize', onManifoldResize);
}

function setupManifoldLighting() {
    // Ambient light
    const ambientLight = new THREE.AmbientLight(0x404060, 0.5);
    manifoldScene.add(ambientLight);
    
    // Directional lights for dramatic effect
    const light1 = new THREE.DirectionalLight(0x00d4ff, 0.8);
    light1.position.set(50, 100, 50);
    manifoldScene.add(light1);
    
    const light2 = new THREE.DirectionalLight(0xff00ff, 0.4);
    light2.position.set(-50, 50, -50);
    manifoldScene.add(light2);
    
    // Point light for glow effect
    const pointLight = new THREE.PointLight(0x00d4ff, 1, 200);
    pointLight.position.set(0, 50, 0);
    manifoldScene.add(pointLight);
}

function setupManifoldControls() {
    // Create orbit controls for mouse interaction
    const OrbitControls = THREE.OrbitControls || window.OrbitControls;
    if (OrbitControls) {
        manifoldControls = new OrbitControls(manifoldCamera, manifoldRenderer.domElement);
        manifoldControls.enableDamping = true;
        manifoldControls.dampingFactor = 0.05;
        manifoldControls.rotateSpeed = 0.5;
        manifoldControls.zoomSpeed = 0.8;
        manifoldControls.panSpeed = 0.8;
    }
}

async function loadManifoldMesh() {
    try {
        const response = await fetch('/api/manifold_mesh');
        const data = await response.json();
        manifoldData = data;
        createManifoldSurface(data);
    } catch (error) {
        console.error('Error loading manifold mesh:', error);
        createDefaultManifold();
    }
}

function createManifoldSurface(data) {
    // Remove existing mesh if any
    if (manifoldMesh) {
        manifoldScene.remove(manifoldMesh);
    }
    
    const geometry = new THREE.BufferGeometry();
    const vertices = [];
    const colors = [];
    const indices = [];
    
    const rows = data.x.length;
    const cols = data.x[0].length;
    
    // Create vertices with color based on metric
    for (let i = 0; i < rows; i++) {
        for (let j = 0; j < cols; j++) {
            // Position (scaled for visualization)
            const x = (data.x[i][j] - 35) * 2;  // Center around c=35
            const y = data.z[i][j];  // Height represents metric
            const z = (data.y[i][j] - 500) * 0.1;  // Wavelength axis
            
            vertices.push(x, y, z);
            
            // Color based on metric value
            const metricValue = data.metric[i][j];
            const color = getMetricColor(metricValue);
            colors.push(color.r, color.g, color.b);
        }
    }
    
    // Create face indices
    for (let i = 0; i < rows - 1; i++) {
        for (let j = 0; j < cols - 1; j++) {
            const a = i * cols + j;
            const b = i * cols + (j + 1);
            const c = (i + 1) * cols + (j + 1);
            const d = (i + 1) * cols + j;
            
            // Two triangles per quad
            indices.push(a, b, d);
            indices.push(b, c, d);
        }
    }
    
    geometry.setAttribute('position', new THREE.Float32BufferAttribute(vertices, 3));
    geometry.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));
    geometry.setIndex(indices);
    geometry.computeVertexNormals();
    
    // Create material with vertex colors
    const material = new THREE.MeshPhongMaterial({
        vertexColors: true,
        side: THREE.DoubleSide,
        transparent: true,
        opacity: 0.8,
        shininess: 100,
        specular: new THREE.Color(0x00d4ff)
    });
    
    // Create mesh
    manifoldMesh = new THREE.Mesh(geometry, material);
    manifoldScene.add(manifoldMesh);
    
    // Add wireframe overlay for structure visibility
    const wireframeGeometry = new THREE.WireframeGeometry(geometry);
    const wireframeMaterial = new THREE.LineBasicMaterial({
        color: 0x00d4ff,
        opacity: 0.2,
        transparent: true
    });
    const wireframe = new THREE.LineSegments(wireframeGeometry, wireframeMaterial);
    manifoldMesh.add(wireframe);
}

function getMetricColor(value) {
    // Create gradient from low (blue) to high (red) metric values
    const normalized = Math.min(1, Math.max(0, (value - 0.1) / 2));
    
    if (normalized < 0.5) {
        // Blue to cyan
        const t = normalized * 2;
        return new THREE.Color(0, t, 1);
    } else {
        // Cyan to magenta
        const t = (normalized - 0.5) * 2;
        return new THREE.Color(t, 1 - t, 1 - t);
    }
}

function createDefaultManifold() {
    // Create a default parametric surface if data loading fails
    const geometry = new THREE.ParametricGeometry(
        (u, v, target) => {
            const x = (u - 0.5) * 100;
            const z = (v - 0.5) * 100;
            const y = Math.sin(u * Math.PI * 2) * Math.cos(v * Math.PI * 2) * 20 + 20;
            target.set(x, y, z);
        },
        50, 50
    );
    
    const material = new THREE.MeshPhongMaterial({
        color: 0x00d4ff,
        wireframe: false,
        transparent: true,
        opacity: 0.7
    });
    
    manifoldMesh = new THREE.Mesh(geometry, material);
    manifoldScene.add(manifoldMesh);
}

function updateManifoldGeodesic(geodesicData) {
    // Clear existing geodesic lines
    geodesicLines.forEach(line => manifoldScene.remove(line));
    geodesicLines = [];
    
    if (!geodesicData || !geodesicData.path) return;
    
    // Create geodesic path
    const points = [];
    const path = geodesicData.path;
    const wavelength = parseFloat(document.getElementById('wavelength').value);
    
    for (let i = 0; i < path.length; i++) {
        const c = path[i][0];
        const v = path[i][1];
        
        // Map to 3D coordinates
        const x = (c - 35) * 2;
        const z = (wavelength - 500) * 0.1;
        
        // Height based on metric at this point
        let y = 20;
        if (manifoldData) {
            // Interpolate metric value
            const metricValue = interpolateMetric(c, wavelength);
            y = metricValue * 20;
        }
        
        points.push(new THREE.Vector3(x, y + 2, z));  // Slight offset above surface
    }
    
    // Create geodesic curve
    const curve = new THREE.CatmullRomCurve3(points);
    const curvePoints = curve.getPoints(100);
    const geometry = new THREE.BufferGeometry().setFromPoints(curvePoints);
    
    // Animated gradient material
    const material = new THREE.LineBasicMaterial({
        color: 0xffd700,  // Gold color for geodesics
        linewidth: 3,
        transparent: true,
        opacity: 0.9
    });
    
    const geodesicLine = new THREE.Line(geometry, material);
    manifoldScene.add(geodesicLine);
    geodesicLines.push(geodesicLine);
    
    // Add glow effect to geodesic
    const glowMaterial = new THREE.LineBasicMaterial({
        color: 0xffd700,
        linewidth: 5,
        transparent: true,
        opacity: 0.3
    });
    const glowLine = new THREE.Line(geometry, glowMaterial);
    manifoldScene.add(glowLine);
    geodesicLines.push(glowLine);
    
    // Add velocity vectors along the path
    addVelocityVectors(geodesicData);
    
    // Animate camera to focus on geodesic
    animateCameraToGeodesic(points);
}

function interpolateMetric(c, lambda) {
    // Simple interpolation for metric value at (c, lambda)
    // In real implementation, this would use the actual metric network
    const c_norm = (c - 30) / 30;
    const lambda_norm = (lambda - 500) / 300;
    
    return 1.0 + 0.5 * Math.exp(-((c_norm - 0.5)**2 + lambda_norm**2) / 0.1);
}

function addVelocityVectors(geodesicData) {
    const path = geodesicData.path;
    const wavelength = parseFloat(document.getElementById('wavelength').value);
    
    // Add velocity arrows at key points
    for (let i = 0; i < path.length; i += 5) {  // Every 5th point
        const c = path[i][0];
        const v = path[i][1];
        
        const origin = new THREE.Vector3(
            (c - 35) * 2,
            interpolateMetric(c, wavelength) * 20 + 2,
            (wavelength - 500) * 0.1
        );
        
        const direction = new THREE.Vector3(v * 0.1, 0, 0).normalize();
        const length = Math.abs(v) * 0.5;
        const color = v > 0 ? 0x00ff00 : 0xff0000;  // Green for positive, red for negative
        
        const arrowHelper = new THREE.ArrowHelper(direction, origin, length, color, length * 0.3);
        manifoldScene.add(arrowHelper);
        geodesicLines.push(arrowHelper);  // Track for cleanup
    }
}

function animateCameraToGeodesic(points) {
    if (!points || points.length === 0) return;
    
    // Calculate bounding box of geodesic
    const box = new THREE.Box3();
    points.forEach(p => box.expandByPoint(p));
    const center = box.getCenter(new THREE.Vector3());
    
    // Smoothly move camera to focus on geodesic
    const targetPosition = new THREE.Vector3(
        center.x + 50,
        center.y + 40,
        center.z + 50
    );
    
    gsap.to(manifoldCamera.position, {
        x: targetPosition.x,
        y: targetPosition.y,
        z: targetPosition.z,
        duration: 1.5,
        ease: "power2.inOut"
    });
    
    gsap.to(manifoldControls.target, {
        x: center.x,
        y: center.y,
        z: center.z,
        duration: 1.5,
        ease: "power2.inOut"
    });
}

function animateManifold() {
    animationFrame = requestAnimationFrame(animateManifold);
    
    // Update controls
    if (manifoldControls) {
        manifoldControls.update();
    }
    
    // Rotate manifold slowly for visual interest
    if (manifoldMesh && !manifoldControls) {
        manifoldMesh.rotation.y += 0.001;
    }
    
    // Animate geodesic glow
    geodesicLines.forEach((line, index) => {
        if (line.material && line.material.opacity !== undefined) {
            const time = Date.now() * 0.001;
            line.material.opacity = 0.5 + Math.sin(time + index) * 0.3;
        }
    });
    
    // Render scene
    manifoldRenderer.render(manifoldScene, manifoldCamera);
}

function onManifoldResize() {
    const container = document.getElementById('manifold-3d');
    const width = container.clientWidth;
    const height = container.clientHeight;
    
    manifoldCamera.aspect = width / height;
    manifoldCamera.updateProjectionMatrix();
    manifoldRenderer.setSize(width, height);
}

// Helper function to create particle effects
function createParticleSystem() {
    const particles = new THREE.BufferGeometry();
    const particleCount = 1000;
    const positions = new Float32Array(particleCount * 3);
    
    for (let i = 0; i < particleCount * 3; i += 3) {
        positions[i] = (Math.random() - 0.5) * 200;
        positions[i + 1] = Math.random() * 100;
        positions[i + 2] = (Math.random() - 0.5) * 200;
    }
    
    particles.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    
    const material = new THREE.PointsMaterial({
        color: 0x00d4ff,
        size: 0.5,
        transparent: true,
        opacity: 0.6,
        blending: THREE.AdditiveBlending
    });
    
    const particleSystem = new THREE.Points(particles, material);
    manifoldScene.add(particleSystem);
    
    // Animate particles
    function animateParticles() {
        particleSystem.rotation.y += 0.0005;
        const positions = particles.attributes.position.array;
        
        for (let i = 1; i < positions.length; i += 3) {
            positions[i] -= 0.1;
            if (positions[i] < 0) {
                positions[i] = 100;
            }
        }
        
        particles.attributes.position.needsUpdate = true;
    }
    
    // Add to animation loop
    const originalAnimate = animateManifold;
    animateManifold = function() {
        animateParticles();
        originalAnimate();
    };
}

// Export for use in other modules
window.updateManifoldGeodesic = updateManifoldGeodesic;