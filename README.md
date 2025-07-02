# fft_cnn
CNN-FFT model compatible with the Outlier Protocol

This service exposes the endpoints defined in `outlier_protocol/protocol.yaml`:

- `GET /health`
- `POST /train`
- `POST /train/{ordinal}`
- `POST /predict`

Set the environment variable `ORCHESTRATOR_URL` so the server can notify the
control plane via the `/trainingCompleted` webhook once training finishes.
