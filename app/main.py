from contextlib import asynccontextmanager

from utils.io import load_model
from utils.graph_constructor import smiles_to_graph 
from pathlib import Path
from torch_geometric.data import Batch
from pipeline.graph_explainer import ModelExplainer 
from utils.molecule_visuzlizer import visualize_2d_quantitative_svg
from io import StringIO

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse
import uvicorn

ml_model = {} 
@asynccontextmanager
async def lifespan(app: FastAPI):

    model_path = Path("trained_model", "trained_model.pth")

    if not model_path.exists(): 
        raise FileNotFoundError(
            f"CRITICAL: Trained model not found at {model_path.resolve()}. "
            "Please ensure the model file is present before starting the server."
        )
    
    model = load_model(model_path) 

    model.eval() 

    ml_model['model'] = model 

    yield 


app = FastAPI(lifespan = lifespan)

app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
def serveRoot():

    html_path = Path("static", "index.html")

    if not Path(html_path).exists():
        return HTTPException(
            status_code=404, detail="Frontend index.html file not found"
        )

    return FileResponse(html_path)


@app.post("/predict")
async def predict_permeability(user_smiles_str: str):

    graph = smiles_to_graph(user_smiles_str, None) 

    if graph is None: 
        return HTTPException(
            status_code= 300, 
            detail = f"Failed to convert smiles to graph: {user_smiles_str}"
        )
    
    model = ml_model['model'] 

    graph = Batch.from_data_list([graph])

    prediction = model.predict(
        graph.x, 
        graph.edge_index, 
        graph.edge_attr,
        graph.batch
    )

    probability = model.predict_probs(
        graph.x, 
        graph.edge_index, 
        graph.edge_attr, 
        graph.batch
    )

    return {'permeable': bool(prediction==1.0), 'probability': float(probability)}


@app.get("/explain")
def explain_prediction(user_smiles_str: str):

    graph = smiles_to_graph(user_smiles_str, None) 

    if graph is None: 
        raise HTTPException(
            status_code= 300, 
            detail = f"Failed to convert smiles to graph: {user_smiles_str}"
        )

    graph = Batch.from_data_list([graph])

    model = ml_model['model'] 

    graph_explainer = ModelExplainer(model) 

    node_scores, edge_scores = graph_explainer.explain_graph(graph)

    explaination_figure = visualize_2d_quantitative_svg(user_smiles_str, node_scores)

    response_buffer = StringIO(explaination_figure) 
    response_buffer.seek(0) 

    return StreamingResponse(response_buffer, media_type= "image/svg+xml")


if __name__ == "__main__":

    uvicorn.run("main:app", 
                host="0.0.0.0", 
                port=7860)
