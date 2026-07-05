let RDKitModule = null;

    // Initialize RDKit.js
    window.initRDKitModule().then(function(instance) {
        RDKitModule = instance;
        console.log("RDKit.js initialized successfully.");
    }).catch((err) => {
        console.error("Failed to load RDKit.js", err);
    });

    async function runPipeline() {
        const smiles = document.getElementById('smilesInput').value.trim();
        if (!smiles) {
            alert("Please enter a valid SMILES string.");
            return;
        }

        // UI Reset & Show Loader
        document.getElementById('submitBtn').disabled = true;
        document.getElementById('mainLoading').style.display = 'block';
        
        try {
            // 1. Local 2D Chemical Structure Rendering
            renderMolecule(smiles);

            // 2. Fetch Prediction from FastAPI
            // Note: Sending as query parameter matching your fastapi signature `user_smiles_str: str`
            const predictResponse = await fetch(`/predict?user_smiles_str=${encodeURIComponent(smiles)}`, {
                method: 'POST'
            });
            
            if (!predictResponse.ok) throw new Error('Prediction API failed');
            const predictData = await predictResponse.json();
            
            // Render Prediction Output
            displayPrediction(predictData);

            // 3. Fetch GNN Explainer Plot
            // Since explainer plots are usually generated per-molecule, you likely want to pass the smiles to /explain too.
            // If your /explain endpoint expects query parameters, adapt the URL below.
            const explainUrl = `/explain?user_smiles_str=${encodeURIComponent(smiles)}`;
            
            const imgElement = document.getElementById('explainerImg');
            const explainPlaceholder = document.getElementById('explainPlaceholder');
            
            imgElement.src = explainUrl;
            imgElement.style.display = 'block';
            explainPlaceholder.style.display = 'none';

            document.getElementById('explainerLegend').style.display = 'block';

        } catch (error) {
            console.error(error);
            document.getElementById('predictionResult').innerHTML = `
                <p style="color:var(--danger)">Error processing request. Check console or API logs.</p>
            `;
        } finally {
            document.getElementById('submitBtn').disabled = false;
            document.getElementById('mainLoading').style.display = 'none';
        }
    }

    function renderMolecule(smiles) {
        const canvas = document.getElementById('molCanvas');
        const placeholder = document.getElementById('molPlaceholder');

        if (!RDKitModule) {
            placeholder.textContent = "RDKit processing...";
            return;
        }

        const mol = RDKitModule.get_mol(smiles);
        if (mol) {
            placeholder.style.display = 'none';
            canvas.style.display = 'block';
            mol.draw_to_canvas(canvas, -1, -1);
            mol.delete(); // Free up WASM memory
        } else {
            canvas.style.display = 'none';
            placeholder.style.display = 'block';
            placeholder.textContent = "Invalid SMILES string - could not parse.";
        }
    }

    function displayPrediction(data) {
        const resultDiv = document.getElementById('predictionResult');
        
        // Adapt this mapping based on exactly what your FastAPI `/predict` returns.
        // Assuming it returns something like: { "permeable": true, "probability": 0.89 }
        const isPermeable = data.permeable; 
        const prob = data.probability !== undefined ? `(probability: ${(data.probability * 100).toFixed(1)}%)` : ''; 
        
        if (isPermeable) {
            resultDiv.innerHTML = `
                <p>The GNN model classifies this molecule as:</p>
                <div class="result-badge permeable">BBB Permeable (+)</div>
                <p style="margin-top:0.5rem; font-size:0.9rem; color:#718096;">${prob}</p>
            `;
        } else {
            resultDiv.innerHTML = `
                <p>The GNN model classifies this molecule as:</p>
                <div class="result-badge non-permeable">Non-Permeable (-)</div>
                <p style="margin-top:0.5rem; font-size:0.9rem; color:#718096;">${prob}</p>
            `;
        }
    }