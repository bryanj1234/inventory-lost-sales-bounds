import sys
import math
import numpy as np

####################################################################################################

print("PYTHON INFO:")
print('=========================================================================')
print(sys.version)
print(sys.executable)
print('=========================================================================')
print("RUNNING:")
print(__file__)
print('=========================================================================')
print()

###################################################################################################

def sample_exponential_interarrival_time(demand_arrival_rate):
    beta = 1/demand_arrival_rate

    return np.random.exponential(beta)


def x_to_k_divided_by_k_factorial(x, k):
    if k == 0:
        return 1

    factors = np.array([x/(k - jj) for jj in range(0, k)])

    return np.prod(factors)


# CALCULATES SUM{k=r}^{infty} (k - r) * x^k/k! e^{-x} IN A BETTER WAY.
def loss_function_1(r, x):
    loss = x - r
    if r >= 1:
        loss += r*math.exp(-x)*x_to_k_divided_by_k_factorial(x, r - 1)
        if r >= 2:
            loss -= (x - r)*math.exp(-x)*np.sum([x_to_k_divided_by_k_factorial(x, k) for k in range(0, r-1)])

    return loss


def calculate_LB_fraction_sales_lost(r, q, lead_time, demand_arrival_rate):
    x = lead_time*demand_arrival_rate
    loss = loss_function_1(r, x)
    LB_fraction_sales_lost = loss / (loss + q*math.floor((r + q)/ q))

    return LB_fraction_sales_lost


# NOTE: THIS IS THE ERLANG B LOSS WHEN q = 1.
def calculate_UB_fraction_sales_lost(r, q, lead_time, demand_arrival_rate):
    x = lead_time*demand_arrival_rate
    v = (r + 1)/(q*math.floor((r + q)/ q))
    numerator = v * x_to_k_divided_by_k_factorial(x, r + 1)
    denominator = numerator + np.sum([x_to_k_divided_by_k_factorial(x, k) for k in range(0, r + 1)])
    UB_fraction_sales_lost = numerator/denominator

    return UB_fraction_sales_lost

# ZIPKIN p.278-279
def Zipkin_fraction_sales_lost_approx(r, q, lead_time, demand_arrival_rate):
    x = lead_time*demand_arrival_rate
    loss = loss_function_1(r, x)
    Zipkin_approx_fraction_sales_lost = loss / (loss + q)

    return Zipkin_approx_fraction_sales_lost

# ZIPKIN p.188
def Backorder_fraction_sales_lost_approx(r, q, lead_time, demand_arrival_rate):
    x = lead_time*demand_arrival_rate
    loss_1 = loss_function_1(r, x)
    loss_2 = loss_function_1(r + q, x)

    return (loss_1 - loss_2)/q


def simulate_lost_sales_system(r, q, lead_time, demand_arrival_rate, num_arrival_epochs, bool_debug=False):
    print('RUNNING simulate_lost_sales_system() using:')
    print('r=' + str(r)
            + ', q=' + str(q)
            + ', demand_arrival_rate=' + str(demand_arrival_rate)
            + ', num_arrival_epochs=' + str(num_arrival_epochs))

    # CALCULATE MAXIMUM POSSIBLE NUMBER OF OUTSTANDING REPLENISHMENT ORDERS.
    max_num_outstanding = math.floor((r + q)/q)

    print("max_num_outstanding:", max_num_outstanding)

    # SEED INVENTORY LEVEL AT REORDER POINT + REORDER QUANTITY
    cur_inv_level = r + q

    # SEED OUTSTANDING ORDERS
    lead_times_outstanding = []

    # EACH ITERATION IS ONE EXPONENTIALLY DISTRIBUTED INTERARRIVAL TIME.
    epoch_num = 0
    num_lost_sales = 0
    while epoch_num < num_arrival_epochs:
        epoch_num += 1

        inventory_position = cur_inv_level + q*len(lead_times_outstanding)
        num_outstanding = len(lead_times_outstanding)

        if bool_debug:
            print()
            print("epoch_num:", epoch_num)
            print("\t", "cur_inv_level:", cur_inv_level)
            print("\t", "inventory_position:", inventory_position)
            print("\t", "num_outstanding:", num_outstanding)
            print("\t", "lead_times_outstanding:", lead_times_outstanding)

        # ALGORTIHM:
        #       NOTE:
        #           THERE IS (UP TO FLOATING POINT LIMITS) PROBABILITY ZERO THAT A REPLENISHMENT ORDER ARRIVES
        #           AT EXACTLY THE SAME EPOCH AS A CUSTOMER ORDER.
        #           THIS MEANS IT DOESN'T MATTER FOR THR SIMULATION WHETHER OR NOT WE
        #           INCREASE INVENTORY LEVEL BEFORE OR AFTER WE DEAL WITH THE DEMAND.
        #   1) CALCULATE NEXT ARRIVAL TIME.
        #   2) REDUCE REMAINING LEAD TIME ON OUTSTANDING ORDERS BY INTERARRIVAL TIME.
        #   3) INCREASE INVENTORY LEVEL FOR DELIVERED ORDERS.
        #   4) REDUCE INVENTORY LEVEL FOR DEMAND ARRIVAL, OR RECORD LOST SALE.

        # 1) CALCULATE NEXT ARRIVAL TIME.
        demand_epoch_delta = sample_exponential_interarrival_time(demand_arrival_rate)

        if bool_debug:
            print("demand_epoch_delta:", demand_epoch_delta)

        # 2) FIXME: REDUCE REMAINING LEAD TIME ON OUTSTANDING ORDERS BY INTERARRIVAL TIME.
        lead_times_outstanding = [lt - demand_epoch_delta for lt in lead_times_outstanding if lt - demand_epoch_delta > 0]
        num_delivered_orders = num_outstanding - len(lead_times_outstanding)
        if bool_debug:
            print("num_delivered_orders:", num_delivered_orders)

        # 3) FIXME: INCREASE INVENTORY LEVEL FOR DELIVERED ORDERS.
        cur_inv_level += q * num_delivered_orders

        # 4) REDUCE INVENTORY LEVEL FOR DEMAND ARRIVAL, OR RECORD LOST SALE.
        if cur_inv_level == 0:      # LOST SALE
            num_lost_sales += 1
        elif cur_inv_level > 0:     # REDUCE INVENTORY, AND MAKE NEW REPLENISHMENT ORDER IF NECESSARY.
            cur_inv_level -= 1
            inventory_position = cur_inv_level + q*len(lead_times_outstanding)
            if inventory_position == r:      # HIT REORDER POINT
                lead_times_outstanding.append(lead_time)
            elif inventory_position < r:
                assert False, "BRYAN HAS A BUG: INVENTORY POSITION SHOULD NEVER BE LESS THAN REORDER POINT."
        else:
            assert False, "BRYAN HAS A BUG: LOST SALES INV LEVEL SHOULD NEVER BE NEGATIVE."


    # CALCULATE FRACTION OF SALES LOST.
    frac_lost_sales = num_lost_sales/num_arrival_epochs

    return num_lost_sales, frac_lost_sales



###################################################################################################
### START CONFIG ##################################################################################
###################################################################################################

r = 20                      # Reorder point
q = 19                       # Reorder quantity
lead_time = 1

demand_arrival_rate = 15    # Poisson arrival rate
num_arrival_epochs = 100000    # Number of demand arrivals to run in the simulation

bool_debug = False

###################################################################################################
### END CONFIG ####################################################################################
###################################################################################################


num_lost_sales, frac_lost_sales = simulate_lost_sales_system(r, q, lead_time, demand_arrival_rate, num_arrival_epochs, bool_debug=bool_debug)



print()
print()
print("#####################################################################")
print("num_arrival_epochs:", num_arrival_epochs)
print("num_lost_sales:", num_lost_sales)
print("frac_lost_sales:", frac_lost_sales)

LB_fraction_sales_lost = calculate_LB_fraction_sales_lost(r, q, lead_time, demand_arrival_rate)
UB_fraction_sales_lost = calculate_UB_fraction_sales_lost(r, q, lead_time, demand_arrival_rate)
Zipkin_approx_fraction_sales_lost = Zipkin_fraction_sales_lost_approx(r, q, lead_time, demand_arrival_rate)
Backorder_approx_fraction_sales_lost = Backorder_fraction_sales_lost_approx(r, q, lead_time, demand_arrival_rate)

print()
print("LB_fraction_sales_lost:", LB_fraction_sales_lost)
print("UB_fraction_sales_lost:", UB_fraction_sales_lost)
print("Zipkin_approx_fraction_sales_lost:", Zipkin_approx_fraction_sales_lost)
print("Backorder_approx_fraction_sales_lost:", Backorder_approx_fraction_sales_lost)