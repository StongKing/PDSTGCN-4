import numpy as np
import torch


def _torch_mask(labels, null_val=0.0):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = labels != null_val
    mask = mask.float()
    mean = mask.mean()
    if mean > 0:
        mask = mask / mean
    return torch.nan_to_num(mask)


def masked_mae(preds, labels, null_val=0.0):
    mask = _torch_mask(labels, null_val)
    return torch.mean(torch.nan_to_num(torch.abs(preds - labels) * mask))


def masked_mse(preds, labels, null_val=0.0):
    mask = _torch_mask(labels, null_val)
    return torch.mean(torch.nan_to_num((preds - labels) ** 2 * mask))


def masked_rmse(preds, labels, null_val=0.0):
    return torch.sqrt(masked_mse(preds, labels, null_val))


def masked_mape_np(y_true, y_pred, null_val=0.0):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if np.isnan(null_val):
        mask = ~np.isnan(y_true)
    else:
        mask = y_true != null_val
    denom = np.abs(y_true)
    valid = mask & (denom > 1e-8)
    if not np.any(valid):
        return np.nan
    return float(np.mean(np.abs(y_pred[valid] - y_true[valid]) / denom[valid]))


def mae_np(y_true, y_pred):
    return float(np.mean(np.abs(np.asarray(y_pred) - np.asarray(y_true))))


def rmse_np(y_true, y_pred):
    d = np.asarray(y_pred) - np.asarray(y_true)
    return float(np.sqrt(np.mean(d * d)))


def pdstgcn_loss(outputs, labels, alpha=1, beta=0.01):
    N = outputs.shape[1]

    loss_pred = torch.mean(
        torch.abs(outputs - labels)
    )

    loss_cons = torch.mean(
        torch.abs(
            outputs.sum(dim=1)
            - labels.sum(dim=1)
        )
    ) / N

    loss_int = torch.mean(
        torch.abs(
            outputs - torch.round(outputs)
        )
    )

    return (
        loss_pred
        + alpha * loss_cons
        + beta * loss_int
    )

def forecasting_loss(
    outputs,
    labels
):

    return torch.mean(
    torch.abs(
        outputs[:, 1:, :]
        -
        labels[:, 1:, :]
    )
)

def pdstgcn2_loss(
    outputs,
    raw_outputs,
    labels,
    target_sum,
    lambda_cons=0.1,
    lambda_adjust=0.05,
    beta_int=0.0,
    return_components=False,
):
    """
    PDSTGCN loss with exact final physical reconciliation.

    Parameters
    ----------
    outputs : Tensor [B,N,H]
        Final physically reconciled prediction.

    raw_outputs : Tensor [B,N,H]
        Backbone prediction BEFORE physical reconciliation.

    labels : Tensor [B,N,H]
        Ground-truth bike inventories.

    target_sum : float
        Closed-system fleet size, e.g. 5310.

    lambda_cons : float
        Weight of raw fleet-conservation loss.

    lambda_adjust : float
        Weight penalizing the size of the physical correction.

    beta_int : float
        Optional integer regularization.
        Recommended to set to 0 initially because final
        integerization is performed only during inference.

    return_components : bool
        If True, also return individual loss components.
    """

    if outputs.shape != labels.shape:
        raise ValueError(
            f"outputs={tuple(outputs.shape)}, "
            f"labels={tuple(labels.shape)}"
        )

    if raw_outputs.shape != outputs.shape:
        raise ValueError(
            f"raw_outputs={tuple(raw_outputs.shape)}, "
            f"outputs={tuple(outputs.shape)}"
        )

    N = outputs.shape[1]

    # ========================================================
    # 1. Final forecasting accuracy
    #
    # Keep this identical to the current main prediction loss
    # so that this experiment changes only the physics training.
    # ========================================================

    loss_pred = torch.mean(
        torch.abs(
            outputs
            -
            labels
        )
    )

    # ========================================================
    # 2. RAW fleet-conservation loss
    #
    # IMPORTANT:
    #
    # Do NOT compute this from outputs because outputs are
    # already exactly conservative.
    #
    # Compute it from raw_outputs.
    #
    # Divide by N so the value has an approximate
    # "bike error per node" scale comparable with MAE.
    # ========================================================

    raw_fleet_error = (
        raw_outputs.sum(
            dim=1
        )
        -
        float(target_sum)
    )                                   # [B,H]

    loss_raw_cons = torch.mean(
        torch.abs(
            raw_fleet_error
        )
    ) / float(N)

    # ========================================================
    # 3. Physical-correction magnitude
    #
    # Encourage the backbone itself to stay close to the
    # physical feasible set, so the reconciliation layer does
    # not need to make large corrections.
    # ========================================================

    loss_adjust = torch.mean(
        torch.abs(
            outputs
            -
            raw_outputs
        )
    )

    # ========================================================
    # 4. Optional integer regularization
    #
    # For the first experiment I recommend beta_int = 0.
    # ========================================================

    if beta_int > 0.0:

        loss_int = torch.mean(
            torch.abs(
                outputs
                -
                torch.round(outputs)
            )
        )

    else:

        loss_int = outputs.new_tensor(
            0.0
        )

    # ========================================================
    # Total loss
    # ========================================================

    total_loss = (
        loss_pred
        +
        lambda_cons
        *
        loss_raw_cons
        +
        lambda_adjust
        *
        loss_adjust
        +
        beta_int
        *
        loss_int
    )

    if return_components:

        components = {
            "pred": loss_pred.detach(),
            "raw_cons": loss_raw_cons.detach(),
            "adjust": loss_adjust.detach(),
            "integer": loss_int.detach(),
        }

        return total_loss, components

    return total_loss