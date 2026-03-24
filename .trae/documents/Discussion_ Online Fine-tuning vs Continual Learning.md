# Discussion Summary: Online Fine-tuning vs. Continual Learning

## Concept Analysis
You are absolutely correct. The current implementation is technically **Online Fine-tuning** (or **Test-Time Adaptation**), rather than full-fledged Continual Learning (CL).

### Why it's Online Fine-tuning:
1.  **Mechanism**: The code in `deploy.py` (`continual_learning_update`) performs a standard SGD/Adam update step based **only on the most recent sample** (`x_t`).
2.  **No Memory**: It lacks core CL components like **Experience Replay (Buffer)** or **Regularization (EWC)** to prevent Catastrophic Forgetting.
3.  **Scope**: It explicitly freezes the RNN cell by default and only updates the `head` layer, which is a classic **Domain Adaptation** strategy to align the output distribution without destroying the feature extractor.

### Why the distinction matters:
*   **Continual Learning**: Usually implies a goal to learn *new* tasks while *preserving* performance on *old* tasks (Stability-Plasticity balance).
*   **Online Fine-tuning**: Focuses on **rapid adaptation** to the *current* data stream. In time-series forecasting with **Concept Drift** (changing patterns), this is often preferred because "forgetting" outdated patterns is actually a feature, not a bug.

## Conclusion
The variable name `continual_learning` is used loosely here. A more precise name for this boolean flag would be `online_adaptation` or `online_finetuning`.

## Next Steps
As per your request, **no code changes** will be made at this stage. We can proceed to:
1.  Discuss further implications of this approach.
2.  Rename the variables later if desired.
3.  Implement a Replay Buffer if you actually need the model to remember historical patterns (True CL).