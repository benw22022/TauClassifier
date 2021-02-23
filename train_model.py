import argparse
import sys


def main(args):
    import numpy as np
    import h5py

    from keras.optimizers import SGD
    from keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger, \
        ReduceLROnPlateau

    from rnn_tauid.models import experimental_model
    from rnn_tauid.utils import load_vars, load_data, train_test_split
    from rnn_tauid.preprocessing import preprocess, save_preprocessing

    # Determine prongness
    if "1p" in args.sig.lower() and "1p" in args.bkg.lower():
        prong = "1p"
    elif "3p" in args.sig.lower() and "3p" in args.bkg.lower():
        prong = "3p"
    else:
        print("Could not infer prongness from sample names.")
        sys.exit(1)

    # Load rules for variables from file or use defaults if None
    jet_vars, trk_vars, cls_vars = load_vars(var_module=args.var_mod, tag=prong)

    # Get variable names and preprocessing functions
    jet_varnames, _, jet_preproc_func = zip(*jet_vars)
    trk_varnames, _, trk_preproc_func = zip(*trk_vars)
    cls_varnames, _, cls_preproc_func = zip(*cls_vars)

    # Load data
    h5file = dict(driver="family", memb_size=8*1024**3)
    with h5py.File(args.sig, "r", **h5file) as sig, \
         h5py.File(args.bkg, "r", **h5file) as bkg:
        lsig = len(sig["TauJets/pt"])
        lbkg = len(bkg["TauJets/pt"])

        if args.fraction:
            sig_idx = int(args.fraction * lsig)
            bkg_idx = int(args.fraction * lbkg)
        else:
            sig_idx = lsig
            bkg_idx = lbkg

        print("Loading sig [:{}] and bkg [:{}]".format(sig_idx, bkg_idx))

        # Load jet data
        jet_data = load_data(sig, bkg, np.s_[:sig_idx], np.s_[:bkg_idx],
                             jet_vars)

        # Load track data
        trk_data = load_data(sig, bkg, np.s_[:sig_idx], np.s_[:bkg_idx],
                             trk_vars, num=args.num_tracks)

        # Load cluster data
        if args.do_clusters:
            cls_data = load_data(sig, bkg, np.s_[:sig_idx], np.s_[:bkg_idx],
                                 cls_vars, num=args.num_clusters)


    # Validation split
    if args.do_clusters:
        jet_train, jet_test, trk_train, trk_test, cls_train, cls_test = \
            train_test_split([jet_data, trk_data, cls_data],
                             test_size=args.test_size)
    else:
        jet_train, jet_test, trk_train, trk_test = train_test_split(
            [jet_data, trk_data], test_size=args.test_size)

    # Apply preprocessing functions
    jet_preproc = preprocess(jet_train, jet_test, jet_preproc_func)
    trk_preproc = preprocess(trk_train, trk_test, trk_preproc_func)
    if args.do_clusters:
        cls_preproc = preprocess(cls_train, cls_test, cls_preproc_func)

    preproc_results = [(jet_varnames, jet_preproc),
                       (trk_varnames, trk_preproc)]
    if args.do_clusters:
        preproc_results.append((cls_varnames, cls_preproc))

    for variables, preprocessing in preproc_results:
        for var, (offset, scale) in zip(variables, preprocessing):
            print(var + ":")
            print("offsets:\n" + str(offset))
            print("scales:\n" + str(scale) + "\n")

    # Save offsets and scales to hdf5 files
    save_kwargs = dict(
        jet_preproc=(jet_varnames, jet_preproc),
        trk_preproc=(trk_varnames, trk_preproc)
    )
    if args.do_clusters:
        save_kwargs["cls_preproc"] = (cls_varnames, cls_preproc)

    save_preprocessing(args.preprocessing, **save_kwargs)

    # Setup training
    shape_jet = jet_train.x.shape[1:]
    shape_trk = trk_train.x.shape[1:]
    if args.do_clusters:
        shape_cls = cls_train.x.shape[1:]
    else:
        shape_cls = None

    model = experimental_model(
        shape_trk, shape_cls, shape_jet,
        dense_units_1_1=args.dense_units_1_1,
        dense_units_1_2=args.dense_units_1_2,
        lstm_units_1_1=args.lstm_units_1_1,
        lstm_units_1_2=args.lstm_units_1_2,
        dense_units_2_1=args.dense_units_2_1,
        dense_units_2_2=args.dense_units_2_2,
        lstm_units_2_1=args.lstm_units_2_1,
        lstm_units_2_2=args.lstm_units_2_2,
        dense_units_3_1=args.dense_units_3_1,
        dense_units_3_2=args.dense_units_3_2,
        dense_units_3_3=args.dense_units_3_3,
        merge_dense_units_1=args.merge_dense_units_1,
        merge_dense_units_2=args.merge_dense_units_2,
        incl_clusters=args.do_clusters)
    model.summary()

    opt = SGD(lr=0.01, momentum=0.9, nesterov=True)
    model.compile(loss="binary_crossentropy", optimizer=opt,
              metrics=["accuracy"])

    # Configure callbacks
    callbacks = []

    early_stopping = EarlyStopping(
        monitor="val_loss", min_delta=0.0001, patience=args.patience, verbose=1)
    callbacks.append(early_stopping)

    model_checkpoint = ModelCheckpoint(
        args.model, monitor="val_loss", save_best_only=True, verbose=1)
    callbacks.append(model_checkpoint)

    if args.csv_log:
        csv_logger = CSVLogger(args.csv_log)
        callbacks.append(csv_logger)

    reduce_lr = ReduceLROnPlateau(patience=4, verbose=1, min_lr=1e-4)
    callbacks.append(reduce_lr)

    # Start training
    if args.do_clusters:
        hist = model.fit(
            [trk_train.x, cls_train.x, jet_train.x], trk_train.y,
            sample_weight=trk_train.w,
            validation_data=([trk_test.x, cls_test.x, jet_test.x],
                             trk_test.y, trk_test.w),
            epochs=args.epochs, batch_size=args.batch_size,
            callbacks=callbacks, verbose=1)
    else:
        hist = model.fit(
            [trk_train.x, jet_train.x], trk_train.y, sample_weight=trk_train.w,
            validation_data=([trk_test.x, jet_test.x], trk_test.y, trk_test.w),
            epochs=args.epochs, batch_size=args.batch_size, callbacks=callbacks,
            verbose=1)

    # Determine best epoch & validation loss
    val_loss, epoch = min(zip(hist.history["val_loss"], hist.epoch))
    print("\nMinimum val_loss {:.5} at epoch {}".format(val_loss, epoch + 1))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("sig", help="Input signal")
    parser.add_argument("bkg", help="Input background")

    parser.add_argument("--preprocessing", default="preproc.h5")
    parser.add_argument("--model", default="model.h5")

    parser.add_argument("--fraction", type=float, default=None)
    parser.add_argument("--num-tracks", type=int, default=10)
    parser.add_argument("--num-clusters", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--csv-log", default=None)
    parser.add_argument("--var-mod", default=None)

    arch = parser.add_argument_group("architecture")
    arch.add_argument("--dense-units-1-1", type=int, default=32)
    arch.add_argument("--dense-units-1-2", type=int, default=32)

    arch.add_argument("--lstm-units-1-1", type=int, default=32)
    arch.add_argument("--lstm-units-1-2", type=int, default=32)

    arch.add_argument("--dense-units-2-1", type=int, default=32)
    arch.add_argument("--dense-units-2-2", type=int, default=32)

    arch.add_argument("--lstm-units-2-1", type=int, default=24)
    arch.add_argument("--lstm-units-2-2", type=int, default=24)


    arch.add_argument("--dense-units-3-1", type=int, default=128)
    arch.add_argument("--dense-units-3-2", type=int, default=128)
    arch.add_argument("--dense-units-3-3", type=int, default= 16)

    arch.add_argument("--merge-dense-units-1", type=int, default=64)
    arch.add_argument("--merge-dense-units-2", type=int, default=32)

    arch.add_argument("--no-clusters", dest="do_clusters", action="store_false")

    args = parser.parse_args()
    main(args)
