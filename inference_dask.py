import matplotlib.pyplot as plt
import torch
import swyft

from forward import *
from dask.distributed import LocalCluster


if __name__ == "__main__":
    # a TMNRE prior function takes a random variable between 0 and 1 and produces an input vector from a prior of your choice

    # nobs
    low = np.array([50.0, 0.0, 0.0])
    high = np.array([100.0, 1.0, 1.0])
    # 1obs 1nobs
    # low = np.array([50.0, 0.0, 0.0, 0.0])
    # high = np.array([100.0, 1.0, 1.0, 1.0])
    # obs

    # low = np.array([50.0, 0.0])
    # high = np.array([100.0, 1.0])
    prior = swyft.get_uniform_prior(low, high)

    observation_o = {'x': np.array([1.0])}

    n_observation_features = observation_o[observation_key].shape[0]
    observation_shapes = {key: value.shape for key, value in observation_o.items()}

    simulator = swyft.DaskSimulator(
        forward,
        n_parameters,
        sim_shapes=observation_shapes
    )

    cluster = LocalCluster(n_workers=5, threads_per_worker=1)
    simulator.set_dask_cluster(cluster)

    # set up storage

    store_dir = '/home/sgagnon/projects/def-jruan/sgagnon/inference/parallel_storage/'

    store = swyft.Store.directory_store(store_dir, simulator=simulator, overwrite=True)
    store.add(n_training_samples, prior)
    store.simulate()

    dataset = swyft.Dataset(n_training_samples, prior, store)
    
    # dataset length list
    dll = []
    dll += [len(dataset)]
    np.savetxt('dll.txt', np.array(dll))

    network_2d = swyft.get_marginal_classifier(
        observation_key=observation_key,
        marginal_indices=marginal_indices_2d,
        observation_shapes=observation_shapes,
        n_parameters=n_parameters,
        hidden_features=32,
        num_blocks=2,
    )
    mre_2d = swyft.MarginalRatioEstimator(
        marginal_indices=marginal_indices_2d,
        network=network_2d,
        device=device,
    )
    mre_2d.train(dataset)

    posterior_2d = swyft.MarginalPosterior(mre_2d, prior, None)
    bound = posterior_2d.truncate(n_posterior_samples_for_truncation, observation_o)

    def do_round_2d(bound, observation_focus, dll):
        store.add(n_training_samples, prior, bound=bound)
        store.simulate()

        dataset = swyft.Dataset(n_training_samples, prior, store, bound = bound)
        dll += [len(dataset)]
        np.savetxt('dll.txt', np.array(dll))

        network_2d = swyft.get_marginal_classifier(
            observation_key=observation_key,
            marginal_indices=marginal_indices_2d,
            observation_shapes=observation_shapes,
            n_parameters=n_parameters,
            hidden_features=32,
            num_blocks=2,
        )
        mre_2d = swyft.MarginalRatioEstimator(
            marginal_indices=marginal_indices_2d,
            network=network_2d,
            device=device,
        )
        mre_2d.train(dataset)

        posterior_2d = swyft.MarginalPosterior(mre_2d, prior, bound)
        new_bound = posterior_2d.truncate(n_posterior_samples_for_truncation, observation_focus)

        return posterior_2d, new_bound

    for i in range(2):
        posterior_2d, bound = do_round_2d(bound, observation_o, dll)

    network_1d = swyft.get_marginal_classifier(
        observation_key=observation_key,
        marginal_indices=marginal_indices_1d,
        observation_shapes=observation_shapes,
        n_parameters=n_parameters,
        hidden_features=32,
        num_blocks=2,
    )
    mre_1d = swyft.MarginalRatioEstimator(
        marginal_indices=marginal_indices_1d,
        network=network_1d,
        device=device,
    )
    mre_1d.train(dataset)

    store.add(n_training_samples + 100, prior, bound=bound)
    store.simulate()
    dataset = swyft.Dataset(n_training_samples, prior, store, bound = bound)

    mre_1d.train(dataset)

    # SAVING

    prior_filename = "dask.prior.pt"
    dataset_filename = "dask.dataset.pt"
    mre_1d_filename = "dask.mre_1d.pt"
    bound_filename = "dask.bound.pt"

    prior.save(prior_filename)
    dataset.save(dataset_filename)
    mre_1d.save(mre_1d_filename)
    bound.save(bound_filename)
