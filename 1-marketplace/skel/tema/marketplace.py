"""
This module represents the Marketplace.

Computer Systems Architecture Course
Assignment 1
March 2021
"""

from concurrent.futures import thread
from queue import Queue
from uuid import uuid4
import logging
from threading import Lock, currentThread

class Marketplace:
    """
    Class that represents the Marketplace. It's the central part of the implementation.
    The producers and consumers use its methods concurrently.
    """
    def __init__(self, queue_size_per_producer):
        """
        Constructor

        :type queue_size_per_producer: Int
        :param queue_size_per_producer: the maximum size of a queue associated with each producer
        """
        self.queue_size_per_producer = queue_size_per_producer
        self.product_lists = {}
        self.carts = {}
        self.publish_lock = Lock()
        

    def register_producer(self):
        """
        Returns an id for the producer that calls this.

        """
        #creating a list for each producer
        id = str(uuid4())
        self.product_lists[id] = []
        logging.info('Registered producer with id: %s', id)
        return id

    def publish(self, producer_id, product):
        """
        Adds the product provided by the producer to the marketplace

        :type producer_id: String
        :param producer_id: producer id

        :type product: Product
        :param product: the Product that will be published in the Marketplace

        :returns True or False. If the caller receives False, it should wait and then try again.
        """
        producer_list = self.product_lists[producer_id]
        if len(producer_list) == self.queue_size_per_producer:
            return False

        producer_list.append(product)       
        return True

    def new_cart(self):
        """
        Creates a new cart for the consumer

        :returns an int representing the cart_id
        """
        #creating a queue for each producer
        id = uuid4()
        self.carts[id.int] = []
        logging.info('Created new cart with id: %d', id.int)
        return id.int

    def add_to_cart(self, cart_id, product):
        """
        Adds a product to the given cart. The method returns

        :type cart_id: Int
        :param cart_id: id cart

        :type product: Product
        :param product: the product to add to cart

        :returns True or False. If the caller receives False, it should wait and then try again
        """

        for prod_id ,list in self.product_lists.items():
            for prod in list:
                if prod == product:
                    self.carts[cart_id].append((product, prod_id))
                    list.remove(product)
                    #print("added product " + str(product.name) + " to cart with id: " + str(cart_id))
                    return True
        
        return False
        

    def remove_from_cart(self, cart_id, product):
        """
        Removes a product from cart.

        :type cart_id: Int
        :param cart_id: id cart

        :type product: Product
        :param product: the product to remove from cart
        """
        for prod, prod_id in self.carts[cart_id]:
            if prod == product:
                # [COULD BE A PROBLEM IF THERE ISN"T SPACE IN PRODUCT LIST]
                self.carts[cart_id].remove((prod, prod_id))
                self.product_lists[prod_id].append(product)
                return
                
        

    def place_order(self, cart_id):
        """
        Return a list with all the products in the cart.

        :type cart_id: Int
        :param cart_id: id cart
        """
        return self.carts[cart_id]
