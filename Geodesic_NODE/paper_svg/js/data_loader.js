// Data loader for arsenic spectral data
async function loadArsenicData() {
    try {
        const response = await fetch('data/arsenic_data.csv');
        const text = await response.text();
        return parseArsenicCSV(text);
    } catch (error) {
        console.error('Error loading data:', error);
        return null;
    }
}

function parseArsenicCSV(csvText) {
    const lines = csvText.trim().split('\n');
    const header = lines[0].split(',');
    
    // Extract concentration values from header (0, 10, 20, 30, 40, 60)
    const concentrations = header.slice(1).map(c => parseFloat(c));
    
    // Parse spectral data
    const wavelengths = [];
    const spectra = {};
    
    // Initialize spectra object for each concentration
    concentrations.forEach(conc => {
        spectra[conc] = [];
    });
    
    // Parse each data row
    for (let i = 1; i < lines.length; i++) {
        const values = lines[i].split(',');
        const wavelength = parseFloat(values[0]);
        wavelengths.push(wavelength);
        
        // Store absorbance values for each concentration
        concentrations.forEach((conc, idx) => {
            spectra[conc].push(parseFloat(values[idx + 1]));
        });
    }
    
    return {
        wavelengths,
        concentrations,
        spectra,
        // Helper function to get spectrum for a specific concentration
        getSpectrum: function(concentration) {
            return this.spectra[concentration] || null;
        },
        // Helper function to get data as array of {wavelength, absorbance} points
        getSpectrumPoints: function(concentration) {
            const spectrum = this.spectra[concentration];
            if (!spectrum) return null;
            
            return this.wavelengths.map((w, i) => ({
                wavelength: w,
                absorbance: spectrum[i]
            }));
        },
        // Get min/max values for scaling
        getAbsorbanceRange: function() {
            let min = Infinity;
            let max = -Infinity;
            
            Object.values(this.spectra).forEach(spectrum => {
                spectrum.forEach(value => {
                    min = Math.min(min, value);
                    max = Math.max(max, value);
                });
            });
            
            return { min, max };
        }
    };
}