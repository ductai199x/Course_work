0. a. [Network Layer (Data Plane)](#network-layer-(data-plane))
   b. [Network Layer (Control Plane)](#network-layer-(control-plane))
1. [IP Addressing](#ip-addressing)
2. [Software-Defined Network](#software-defined-network)
3. [ICMP](#icmp)
4. [VLAN](#vlan)
5. [MPLS](#mpls)
6. [Data Center Networking](#data-center-networking)
7. [Wireless and Mobile Networks](#wireless-and-mobile-networks)


# Network Layer (Data Plane)
## Longest Prefix Matching
## IP Datagram Format
## IP Fragmentation
## IP Addressing: CIDR
## Subneting
## DHCP
## NAT
## IPv6
## Tunneling

# Network Layer (Control Plane)
## Routing Algorithm Classification
## Link-State (Dijkstra)
## Distance Vector (Bellman-Ford)
## Inter-Autonomous Systems Routing - BGP
## Intra-Autonomous Systems Routing - OSPF

# Sofware-Defined Network
## Why SDN?
- Internet network layer: historically has been implemented via distributed, **per-router** approach
- Monolithic router contains switching hardware, runs ***proprietary*** implementation of Internet standard protocols (IP, RIP, IS-IS, OSPF, BGP) in proprietary router OS (e.g., Cisco IOS) => routers need different **middleboxes** for different network layer functions: firewalls, load balancers, NAT boxes, ..

## (OpenFlow) Generalized Forwarding
- Each router contains a flow table that is computed and distributed by a logically centralized routing controller
- *Logically Centralized Controller*: A distinct (typically remote) controller interacts with local control agents (CAs) in routers to compute forwarding tables
- Why LCC?
    - Easier network management: avoid router misconfigurations, greater flexibility of traffic flows
    - Table-based forwarding (OpenFlow API) allows programmable routers
    - Centralized "programming" easier: compute tables centrally and distribute
    - Distributed "programming": more difficult: compute tables as result of distributed algorithm (protocol) implemented in each and every router 
    - Open (non-proprietary) implementation of control plane


## (OpenFlow) Data Plane Abstraction
- Flow: defined by header fields
- Generalized forwarding: simple packet-handling rules
- Pattern: match values in packet header fields
- Actions: for matched packet: drop, forward, modify, matched packet or send matched packet to controller 
- Priority: disambiguate overlapping patterns
- Counters: #bytes and #packets

## (OpenFlow) Flow Table Entries
- Flow table in a router (computed and distributed by controller) define router’s match+action rules
- Actions:
    - Forward packet to port(s)
    - Encapsulate and forward to controller
    - Drop packet
    - Send to normal processing pipeline
    - Modify Fields

# ICMP
- Used by hosts & routers to communicate network-level information
- Error reporting: unreachable host, network, port, protocol
echo request/reply (used by ping)
- Network-layer “above” IP:
    - ICMP messages carried in IP datagrams
    - ICMP message: type, code plus first 8 bytes of IP datagram causing error

## Traceroute
*Need to include info from Wireshark lab*
- Source sends series of UDP segments to destination: first set has TTL =1 second set has TTL=2, etc. with unlikely port number
- When datagram in nth set arrives to nth router:
    - Router discards datagram and sends back to source ICMP message (type 11, code 0)
    - ICMP message include name of router & IP address
- When ICMP message arrives, source records RTTs
- UDP segment eventually arrives at destination host
destination returns ICMP “port unreachable” message (type 3, code 3) => source stops

# VLAN
- Switch(es) supporting VLAN capabilities can be configured to define multiple virtual LANS over single physical LAN infrastructure.

## Port-based VLAN
- Port-based VLAN: switch ports grouped (by switch management software) so that single physical switch operates as multiple virtual switches
- Traffic isolation: frames to/from ports 1-8 can only reach ports 1-8
- Can also define VLAN based on MAC addresses of endpoints, rather than switch port
- Dynamic membership: ports can be dynamically assigned among VLANs
- Forwarding between VLANS: done via routing (just as with separate switches)
- Trunk port: carries frames between VLANS defined over multiple physical switches:
    - Frames forwarded within VLAN between switches can’t be vanilla 802.1 frames (must carry VLAN ID info) 
    - 802.1q protocol adds/removed additional header fields for frames forwarded between trunk ports

# MPLS
- Initial goal: high-speed IP forwarding using fixed length label (instead of IP address) 
- Fast lookup using fixed length identifier (rather than shortest prefix matching)
- Flexibility:  MPLS forwarding decisions can differ from those of IP
    - Use destination and source addresses to route flows to same destination differently (traffic engineering)
    - Re-route flows quickly if link fails: pre-computed backup paths (useful for VoIP)

*Need to look at MPLS handout*

# Data Center Networking
- Contains hundred of thousands of hosts. These hosts perform massively distributed operations. They are stacked on a rack. At the top of each rack is a switch called TOR (Top Of Rack) that interconnects the hosts with each other and with other switches in the data center.
- Traffic flows:
    - Between external clients and internal hosts (use **border router** )
    - Between internal hosts

## Hierachy Architecture
- Border router -> Access routers -> Top tier switches -> Multiple second tier switches + load balancer -> TOR switches.

## Load balancing
- External requests will first be sent to the load balancer. The load balancer will distribute the load among the hosts to prevent overwhelming a subset of hosts (cause slow processing for certain users)
- Also provide NAT functions, translating public IP address to internal IP address of hosts, then translating back in reverse direction to the client (security)

## Traffic - limited host-to-host capacity problem
- Consider a traffic pattern consisting of 40 simultaneous flows between 40 pairs of hosts in different racks. 
- Specifically, suppose each of 10 hosts in rack 1 sends a flow to a corresponding host in rack 5. Similarly, there are ten simultaneous flows between pairs of hosts in racks 2 and 6, ten simultaneous flows between racks 3 and 7, and ten simultaneous flows between racks 4 and 8. 
- If each flow evenly shares a link’s capacity with other flows traversing that link, then the 40 flows crossing the 10 Gbps A-to-B link (as well as the 10 Gbps B-to-C link) will each only receive 10 Gbps / 40 = 250 Mbps, which is significantly less than the 1 Gbps network interface card rate. The problem becomes even more acute for flows between hosts that need to travel higher up the hierarchy. One possible solution to this limitation is to deploy higher-rate switches and routers. But this would significantly increase the cost of the data center, because switches and routers with high port speeds are very expensive.

## Fully Connected Topology
- In this design, each tier-1 switch connects to all of the tier-2 switches so that (1) host-to-host traffic never has to rise above the switch tiers, and (2) with n tier-1 switches, between any two tier-2 switches there are n disjoint paths.

# Wireless and Mobile Networks
- Wireless network includes:
    - Wireless hosts
    - Base stations
    - Wireless link
    - Infrastructure Mode 
    - *Ad hoc mode* (no base stations, can only transmit to other nodes within link coverage, nodes organize themselves into a network: route among themselves)

## Characteristics
- Decreased signal strength: radio signal attenuates as it propagates through matter (path loss)
- Interference from other sources: standardized wireless network frequencies (e.g., 2.4 GHz) shared by other devices (e.g., phone); devices (motors) interfere as well
- Multipath propagation: radio signal reflects off objects ground, arriving at destination at slightly different times

### Hidden terminal problem
- B, A hear each other
- B, C hear each other
- A, C can't hear each other. A & C unaware of their inteference at B

## CDMA
- Each bit being sent is encoded by multiplying the bit by a signal (the code) that changes at a much faster rate (known as the *chipping rate*) than the original sequence of data bits
> $$Z_{i, m} = d_i \cdot c_m $$
- In a simple world, with no interfering senders, the receiver would receive the encoded bits, $Z_{i,m}$, and recover the original data bits, $d_i$, by computing:
> $$d_i = \frac{1}{M} \sum_{m=1}^{M}Z_{i,m} \cdot c_m $$
- In real world, when the data bits are tangled with bits by other senders. Sender $s$ computes its transmission $Z_{i,m}^s$. Hence, the aggregated received signal is:
> $$Z_{i, m}^* = \sum_{s=1}^{N}Z_{i,m}^s $$
> $$d_i = \frac{1}{M} \sum_{m=1}^{M}Z_{i,m}^* \cdot c_m $$

## CSMA/CA
- 802.11 sender:
    - 1 . if sense channel idle for DIFS then transmit entire frame (no CD)
    - 2 . if sense channel busy then start random backoff time timer -> counts down while channel idle -> transmit when timer expires. If no ACK, increase random backoff interval, repeat step 2
- 802.11 receiver:
    - if frame received OK -> return ACK after SIFS (ACK needed due to hidden terminal problem) 

### Protocol
1. If initially the station senses the channel idle, it transmits its frame after a short period of time known as the Distributed Inter-frame Space (DIFS)
2. Otherwise, the station chooses a random backoff value using binary exponential backoff (as we encountered in  Section 5.3.2) and counts down this value when the channel is sensed idle. While the channel is sensed busy, the counter value remains frozen
3. When the counter reaches zero (note that this can only occur while the channel is sensed idle), the station transmits the entire frame and then waits for an acknowledgment
4. If an acknowledgment is received, the transmitting station knows that its frame has been correctly received at the destination station. If the station has another frame to send, it begins the CSMA/CA protocol at step 2. If the acknowledgment isn’t received, the transmitting station reenters the backoff phase in step 2, with the random value chosen from a larger interval