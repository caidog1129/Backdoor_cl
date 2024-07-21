import numpy as np
import logging
import os

def generate(filename, seed=1, n_items=100, n_bids=500, min_value=1, max_value=100,
                       value_deviation=0.5, add_item_prob=0.9, max_n_sub_bids=5,
                       additivity=0.2, budget_factor=1.5, resale_factor=0.5,
                       integers=False):
    """
    Generate a Combinatorial Auction problem following the 'arbitrary' scheme found in section 4.3. of
        Kevin Leyton-Brown, Mark Pearson, and Yoav Shoham. (2000).
        Towards a universal test suite for combinatorial auction algorithms.
        Proceedings of ACM Conference on Electronic Commerce (EC-00) 66-76.

bibtex
@inproceedings{leyton2000towards,
  title={Towards a universal test suite for combinatorial auction algorithms},
  author={Leyton-Brown, Kevin and Pearson, Mark and Shoham, Yoav},
  booktitle={Proceedings of the 2nd ACM conference on Electronic commerce},
  pages={66--76},
  year={2000}
}


    Saves it as a CPLEX LP file.

    Parameters
    ----------
    filename : str
        Path to the file to save.
    seed : int
        A random seed to use for generating the instance
    n_items : int
        The number of items.
    n_bids : int
        The number of bids.
    min_value : int
        The minimum resale value for an item.
    max_value : int
        The maximum resale value for an item.
    value_deviation : int
        The deviation allowed for each bidder's private value of an item, relative from max_value.
    add_item_prob : float in [0, 1]
        The probability of adding a new item to an existing bundle.
    max_n_sub_bids : int
        The maximum number of substitutable bids per bidder (+1 gives the maximum number of bids per bidder).
    additivity : float
        Additivity parameter for bundle prices. Note that additivity < 0 gives sub-additive bids, while additivity > 0 gives super-additive bids.
    budget_factor : float
        The budget factor for each bidder, relative to their initial bid's price.
    resale_factor : float
        The resale factor for each bidder, relative to their initial bid's resale value.
    integers : logical
        Should bid's prices be integral ?
    """

    assert min_value >= 0 and max_value >= min_value
    assert add_item_prob >= 0 and add_item_prob <= 1
    random = np.random.RandomState(seed)
    def choose_next_item(bundle_mask, interests, compats, add_item_prob, random):
        n_items = len(interests)
        prob = (1 - bundle_mask) * interests * compats[bundle_mask, :].mean(axis=0)
        prob /= prob.sum()
        return random.choice(n_items, p=prob)

    # common item values (resale price)
    values = min_value + (max_value - min_value) * random.rand(n_items)

    # item compatibilities
    compats = np.triu(random.rand(n_items, n_items), k=1)
    compats = compats + compats.transpose()
    compats = compats / compats.sum(1)

    bids = []
    n_dummy_items = 0

    # create bids, one bidder at a time
    while len(bids) < n_bids:

        # bidder item values (buy price) and interests
        private_interests = random.rand(n_items)
        private_values = values + max_value * value_deviation * (2 * private_interests - 1)

        # substitutable bids of this bidder
        bidder_bids = {}

        # generate initial bundle, choose first item according to bidder interests
        prob = private_interests / private_interests.sum()
        item = random.choice(n_items, p=prob)
        bundle_mask = np.full(n_items, 0)
        bundle_mask[item] = 1

        # add additional items, according to bidder interests and item compatibilities
        while random.rand() < add_item_prob:
            # stop when bundle full (no item left)
            if bundle_mask.sum() == n_items:
                break
            item = choose_next_item(bundle_mask, private_interests, compats, add_item_prob, random)
            bundle_mask[item] = 1

        bundle = np.nonzero(bundle_mask)[0]

        # compute bundle price with value additivity
        price = private_values[bundle].sum() + np.power(len(bundle), 1 + additivity)
        if integers:
            price = int(price)

        # drop negativaly priced bundles
        if price < 0:
            logging.debug('negatively priced bundle avoided')
            continue

        # bid on initial bundle
        bidder_bids[frozenset(bundle)] = price

        # generate candidates substitutable bundles
        sub_candidates = []
        for item in bundle:

            # at least one item must be shared with initial bundle
            bundle_mask = np.full(n_items, 0)
            bundle_mask[item] = 1

            # add additional items, according to bidder interests and item compatibilities
            while bundle_mask.sum() < len(bundle):
                item = choose_next_item(bundle_mask, private_interests, compats, add_item_prob, random)
                bundle_mask[item] = 1

            sub_bundle = np.nonzero(bundle_mask)[0]

            # compute bundle price with value additivity
            sub_price = private_values[sub_bundle].sum() + np.power(len(sub_bundle), 1 + additivity)
            if integers:
                sub_price = int(sub_price)

            sub_candidates.append((sub_bundle, sub_price))

        # filter valid candidates, higher priced candidates first
        budget = budget_factor * price
        min_resale_value = resale_factor * values[bundle].sum()
        for bundle, price in [
            sub_candidates[i] for i in np.argsort([-price for bundle, price in sub_candidates])]:

            if len(bidder_bids) >= max_n_sub_bids + 1 or len(bids) + len(bidder_bids) >= n_bids:
                break

            if price < 0:
                logging.debug('negatively priced substitutable bundle avoided')
                continue

            if price > budget:
                logging.debug('over priced substitutable bundle avoided')
                continue

            if values[bundle].sum() < min_resale_value:
                logging.debug('substitutable bundle below min resale value avoided')
                continue

            if frozenset(bundle) in bidder_bids:
                logging.debug('duplicated substitutable bundle avoided')
                continue

            bidder_bids[frozenset(bundle)] = price

        # add XOR constraint if needed (dummy item)
        if len(bidder_bids) > 2:
            dummy_item = [n_items + n_dummy_items]
            n_dummy_items += 1
        else:
            dummy_item = []

        # place bids
        for bundle, price in bidder_bids.items():
            bids.append((list(bundle) + dummy_item, price))

    # generate the LP file
    with open(filename, 'w') as file:
        bids_per_item = [[] for item in range(n_items + n_dummy_items)]

        file.write("maximize\nOBJ:")
        for i, bid in enumerate(bids):
            bundle, price = bid
            file.write(f" +{price} x{i+1}")
            for item in bundle:
                bids_per_item[item].append(i)

        file.write("\n\nsubject to\n")
        for item_bids in bids_per_item:
            if item_bids:
                for i in item_bids:
                    file.write(f" +1 x{i+1}")
                file.write(f" <= 1\n")

        file.write("\nbinary\n")
        for i in range(len(bids)):
            file.write(f" x{i+1}")

def main():
    # 1. Create a folder called 'ca'
    if not os.path.exists('ca'):
        os.mkdir('ca')

    # # 2. Create a subfolder called 'train' and call the generate function 200 times
    # train_folder = os.path.join('ca', 'train')
    # if not os.path.exists(train_folder):
    #     os.mkdir(train_folder)

    # for i in range(1, 201):
    #     filename = os.path.join(train_folder, f'ca_{i}.lp')
    #     generate(filename, seed=i+10000, n_items=200, n_bids=1000)

    # 3. Create a subfolder called 'test' and call the generate function 100 times
    test_folder = os.path.join('ca_h', 'train')
    if not os.path.exists(test_folder):
        os.mkdir(test_folder)

    for i in range(1, 201):
        filename = os.path.join(test_folder, f'ca_{i}.lp')
        generate(filename, seed=i+53, n_items=200, n_bids=1000)

if __name__ == '__main__':
    main()