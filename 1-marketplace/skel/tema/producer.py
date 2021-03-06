"""
This module represents the Producer.

Computer Systems Architecture Course
Assignment 1
March 2021
"""

from itertools import product
from pickletools import markobject
from threading import Thread, Lock
from time import sleep


class Producer(Thread):
    """
    Class that represents a producer.
    """

    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        """
        Constructor.

        @type products: List()
        @param products: a list of products that the producer will produce

        @type marketplace: Marketplace
        @param marketplace: a reference to the marketplace

        @type republish_wait_time: Time
        @param republish_wait_time: the number of seconds that a producer must
        wait until the marketplace becomes available

        @type kwargs:
        @param kwargs: other arguments that are passed to the Thread's __init__()
        """

        Thread.__init__(self,  **kwargs)
        self.products = products
        self.marketplace = marketplace
        self.republish_wait_time = republish_wait_time
        self.id = marketplace.register_producer()
        

    def run(self):
        while True:
            #produce all products
            for product, quantity, wait_time in self.products:
                for i in range(quantity):
                    publish_success = False
                    while publish_success is False:
                        publish_success = self.marketplace.publish(self.id, product)
                        if publish_success is False:
                            sleep(self.republish_wait_time)
                    sleep(wait_time)
