import simpy
import numpy as np
import matplotlib.pyplot as plt

# Set a random seed to ensure reproducible results.
np.random.seed(42)

#Parameter setting
lambda_0 = 550  # Basic arrival rate (per day)
mu = 32         # Service rate (per day）
S = 14          # Number of service counters
total_days = 30 # Simulated total days (increased to ensure stability)
hours_per_day = 24
total_hours = total_days * hours_per_day
r_values = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]  # Cancellation Rate List
warmup_days = 1  # Number of warm-up days
warmup_hours = warmup_days * hours_per_day

# The new queue affects the function parameters
l_0 = 200  # The threshold of queue length at which the impact begins to be significant
k = 0.2   # Steepness of the control function
a = 0.5   # Maximum Impact Factor

def is_weekend(t):
    """Determine if the current time (in hours) is a weekend."""
    day_of_week = int(t // hours_per_day) % 7  # 0 represents Monday, 6 represents Sunday.
    return day_of_week == 5 or day_of_week == 6  # Saturday or Sunday

def queue_impact(l, l_0=20, k=0.2, a=0.5):
    """The impact of calculating queue length on arrival rate"""
    return 1 - (1 / (1 + np.exp(-k * (l - l_0)))) * a

def customer(env, servers, wait_times, service_times):
    """The process of customers arriving and requesting service"""
    arrival_time = env.now
    with servers.request() as request:
        yield request  # Waiting for service desk available
        wait_time = env.now - arrival_time
        wait_times.append(wait_time)
        
        # If it is a weekend, reduce the service rate or delay the start of service.
        while is_weekend(env.now):
            yield env.timeout(1)  # Check once an hour.
        # Start service
        mu_hour = mu / hours_per_day  # Service rate per hour
        service_time = np.random.exponential(1 / mu_hour)
        service_times.append(service_time)
        yield env.timeout(service_time)

def customer_arrivals(env, servers, r, arrival_rates, lambda_0, l_0, k, a):
    """Customer arrival process, considering the nonlinear impact of queue length and adjusting the weekend arrival rate."""
    while True:
        # Get the current queue length
        l = len(servers.queue) + servers.count  # Number of customers in the current system

        # Determine if it is the weekend
        if is_weekend(env.now):
            weekend_factor = 0.2  # Weekend arrival rate reduced by half
        else:
            weekend_factor = 1.0

        # Calculate the current arrival rate, using the new queue impact function and cancellation rate.
        lambda_l = lambda_0 * queue_impact(l, l_0, k, a) * (1 - r) * weekend_factor
        lambda_l_hour = lambda_l / hours_per_day

        # Prevent arrival rate from being zero.
        if lambda_l_hour <= 0:
            interarrival_time = 1e6  # A large time interval, actually will not occur.
        else:
            interarrival_time = np.random.exponential(1 / lambda_l_hour)
        
        yield env.timeout(interarrival_time)

        # New customers arrived
        env.process(customer(env, servers, arrival_rates['wait_times'], arrival_rates['service_times']))

def monitor(env, servers, times, queue_lengths, utilization_tracker, service_times):
    """Monitor queue length and service desk utilization."""
    while True:
        times.append(env.now)
        queue_lengths.append(len(servers.queue) + servers.count)
        # Count the number of customers currently being served
        utilization_tracker.append(servers.count / S)
        yield env.timeout(1)  # Record once per hour.

def simulate_queue(r, collect_data_after=0):
    """Run simulation and return monitoring data"""
    env = simpy.Environment()
    servers = simpy.Resource(env, capacity=S)
    
   # Used for recording data
    times = []
    queue_lengths = []
    wait_times = []
    service_times = []
    utilization = []

   # Used for tracking utilization
    utilization_tracker = []

    # Start customer arrival and monitoring process
    env.process(customer_arrivals(env, servers, r, {'wait_times': wait_times, 'service_times': service_times}, 
                                  lambda_0, l_0, k, a))
    env.process(monitor(env, servers, times, queue_lengths, utilization_tracker, service_times))

    # Warm-up period
    env.run(until=warmup_hours)

    # Reset monitoring data
    times = []
    queue_lengths = []
    utilization_tracker = []
    wait_times = []
    service_times = []

    # Start a new monitoring process
    env.process(customer_arrivals(env, servers, r, {'wait_times': wait_times, 'service_times': service_times}, 
                                  lambda_0, l_0, k, a))
    env.process(monitor(env, servers, times, queue_lengths, utilization_tracker, service_times))

    # Remaining simulation time to run
    env.run(until=total_hours)

    # Calculate utilization rate
    average_utilization = np.mean(utilization_tracker)

    # Calculate average waiting time and service time
    average_wait_time = np.mean(wait_times) if wait_times else 0
    average_service_time = np.mean(service_times) if service_times else 0

    return {
        'times': times,
        'queue_lengths': queue_lengths,
        'average_utilization': average_utilization,
        'average_wait_time': average_wait_time,
        'average_service_time': average_service_time
    }

# Visualization results
plt.figure(figsize=(15, 8))

for r in r_values:
    results = simulate_queue(r)
    times_days = np.array(results['times']) / hours_per_day
    plt.plot(times_days, results['queue_lengths'], label=f'r = {r}')
    print(f"online ratio r = {r}:")
    print(f"  Average queue length: {np.mean(results['queue_lengths']):.2f}")
    print(f"  Average waiting time: {results['average_wait_time']:.2f} h")
    print(f"  Average service time: {results['average_service_time']:.2f} h")
    print(f"  Average utilization rate: {results['average_utilization']*100:.2f}%\n")

plt.xlabel('Days',fontsize=16)
plt.ylabel('Queue Length',fontsize=16)
plt.title('Changes in queue length under different online ratio',fontsize=20)
plt.legend()
plt.grid(True)
plt.savefig('Qlength.png', dpi=300)
plt.show()

# Visualization utilization
plt.figure(figsize=(15, 6))
average_utilizations = []
for r in r_values:
    results = simulate_queue(r)
    average_utilizations.append(results['average_utilization'])
    print(f"online ratio r = {r}: Average utilization rate = {results['average_utilization']*100:.2f}%")

plt.bar([f'r = {r}' for r in r_values], [u * 100 for u in average_utilizations], color='skyblue')
plt.xlabel('online ratio r')
plt.ylabel('Average utilization rate (%)')
plt.title('Average Service Desk Utilization at Different online Ratio')
plt.grid(axis='y')
plt.show()