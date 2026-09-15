import tensorflow as tf

def get_callbacks(patience_early_stop=5, patience_reduce_lr=3, reduce_lr_factor=0.5):

    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=patience_early_stop,
        restore_best_weights=True,
        verbose=0
    )
    
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=reduce_lr_factor,
        patience=patience_reduce_lr,
        min_lr=1e-7,
        verbose=0
    )
    
    return [early_stop, reduce_lr]