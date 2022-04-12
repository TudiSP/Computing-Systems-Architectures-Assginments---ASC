"""
This module represents the Consumer.

Computer Systems Architecture Course
Assignment 1
March 2021
"""

from threading import Thread, currentThread
from time import sleep


class Consumer(Thread):
    """
    Class that represents a consumer.
    """

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        """
        Constructor.

        :type carts: List
        :param carts: a list of add and remove operations

        :type marketplace: Marketplace
        :param marketplace: a reference to the marketplace

        :type retry_wait_time: Time
        :param retry_wait_time: the number of seconds that a producer must wait
        until the Marketplace becomes available

        :type kwargs:
        :param kwargs: other arguments that are passed to the Thread's __init__()
        """
        Thread.__init__(self,  **kwargs)
        self.marketplace = marketplace
        self.retry_wait_time = retry_wait_time
        self.carts = {}

        for cart in carts:
            cart_id = marketplace.new_cart()
            self.carts[cart_id] = cart 

    def run(self):
        for cart_id in self.carts:
            for operation in self.carts[cart_id]:
                quantity = operation['quantity']
                for i in range(quantity):
                    if operation['type'] == 'add':
                        add_success = False
                        while add_success is False:
                            add_success = self.marketplace.add_to_cart(cart_id, operation['product'])
                            if add_success is False:
                                sleep(self.retry_wait_time)
                    if operation['type'] == 'remove':
                        self.marketplace.remove_from_cart(cart_id, operation['product'])

            for product, _ in self.marketplace.place_order(cart_id):
                print(self.getName(), "bought",  product)
                    