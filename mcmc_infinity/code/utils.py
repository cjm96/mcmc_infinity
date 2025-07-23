import jax
import jax.numpy as jnp


def logit(x, bounds=None):
    """
    Logit function with optional bounds transformation.
    
    INPUTS
    ------
    x : jnp.ndarray
        Input array. Values should be in (0, 1) if bounds is None,
        or within the specified bounds.
    bounds : jnp.ndarray, optional
        Array of shape (..., 2) specifying [lower, upper] bounds.
        If provided, x is first scaled to (0, 1) before logit transform.
    
    RETURNS
    -------
    y : jnp.ndarray
        Logit-transformed values.
    log_j : jnp.ndarray
        Log Jacobian determinant of the transformation.
    """
    x = jnp.asarray(x)
    
    # Initialize log Jacobian with consistent shape
    if x.ndim == 1:
        log_j = 0.0
    else:
        log_j = jnp.zeros(x.shape[0])
    
    if bounds is not None:
        bounds = jnp.asarray(bounds)
        # Validate bounds
        if bounds.shape[-1] != 2:
            raise ValueError("bounds must have shape (..., 2)")
            
        # Scale to (0, 1)
        x = (x - bounds[..., 0]) / (bounds[..., 1] - bounds[..., 0])
        
        # Add log Jacobian for the scaling
        bound_range = bounds[..., 1] - bounds[..., 0]
        if x.ndim == 1:
            log_j = log_j - jnp.sum(jnp.log(bound_range))
        else:
            log_j = log_j - jnp.sum(jnp.log(bound_range), axis=-1)
    
    # Clamp to avoid numerical issues
    eps = 1e-7
    x = jnp.clip(x, eps, 1 - eps)
    
    # Compute logit and its Jacobian
    logit_jacobian = -jnp.log(x) - jnp.log1p(-x)
    if x.ndim == 1:
        log_j = log_j + jnp.sum(logit_jacobian)
    else:
        log_j = log_j + jnp.sum(logit_jacobian, axis=-1)
    
    # Apply logit transformation
    y = jnp.log(x / (1 - x))
    
    return y, log_j


def inv_logit(x, bounds=None):
    """
    Inverse logit function with optional bounds transformation.
    
    INPUTS
    ------
    x : jnp.ndarray
        Input array (logit space).
    bounds : jnp.ndarray, optional
        Array of shape (..., 2) specifying [lower, upper] bounds.
        If provided, output is scaled from (0, 1) to the bounds.
    
    RETURNS
    -------
    y : jnp.ndarray
        Inverse logit-transformed values.
    log_j : jnp.ndarray
        Log Jacobian determinant of the transformation.
    """
    x = jnp.asarray(x)
    
    # Apply inverse logit (sigmoid)
    y = jax.nn.sigmoid(x)  # More numerically stable than 1/(1+exp(-x))
    
    # Compute log Jacobian of inverse logit
    inv_logit_jacobian = jnp.log(y) + jnp.log1p(-y)
    if x.ndim == 1:
        log_j = jnp.sum(inv_logit_jacobian)
    else:
        log_j = jnp.sum(inv_logit_jacobian, axis=-1)
    
    if bounds is not None:
        bounds = jnp.asarray(bounds)
        # Validate bounds
        if bounds.shape[-1] != 2:
            raise ValueError("bounds must have shape (..., 2)")
            
        # Scale from (0, 1) to bounds
        bound_range = bounds[..., 1] - bounds[..., 0]
        y = y * bound_range + bounds[..., 0]
        
        # Add log Jacobian for the scaling
        if x.ndim == 1:
            log_j = log_j + jnp.sum(jnp.log(bound_range))
        else:
            log_j = log_j + jnp.sum(jnp.log(bound_range), axis=-1)
    
    return y, log_j
