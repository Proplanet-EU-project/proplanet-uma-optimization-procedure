from fastapi import FastAPI
from routers import optimization_procedure

app = FastAPI(
    title="Optimization Procedure API",
    description="UMA's optimization procedure API",
    version="1.0"
)

app.include_router(optimization_procedure.router)

if __name__ == '__main__':
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)