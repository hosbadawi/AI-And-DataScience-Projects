/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License version 2 as
 * published by the Free Software Foundation;
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
 *
 * Author: Junling Bu <linlinjavaer@gmail.com>
 */
#include "ns3/command-line.h"
#include "ns3/node.h"
#include "ns3/packet.h"
#include "ns3/simulator.h"
#include "ns3/node-container.h"
#include "ns3/net-device-container.h"
#include "ns3/yans-wifi-helper.h"
#include "ns3/mobility-helper.h"
#include "ns3/seq-ts-header.h"
#include "ns3/wave-net-device.h"
#include "ns3/wave-mac-helper.h"
#include "ns3/wave-helper.h"

#include "ns3/random-walk-2d-mobility-model.h"
#include "ns3/position-allocator.h"
#include "ns3/string.h"
#include "ns3/wifi-mode.h"
#include <string>


using namespace ns3;
/**
 * This simulation is to show the routing service of WaveNetDevice described in IEEE 09.4.
 *
 * note: although txPowerLevel is supported now, the "TxPowerLevels"
 * attribute of YansWifiPhy is 1 which means phy devices only support 1
 * levels. Thus, if users want to control txPowerLevel, they should set
 * these attributes of YansWifiPhy by themselves..
 */
class WaveNetDeviceExample
{
public:
  /// Send WSMP example function
  void SendWsmpExample (void);


private:

  /**
   * Send one WSMP packet function
   * \param Channel the channel to use
   * \param Seq the sequence
   */
  void SendOneWsmpPacket (uint32_t Channel,uint32_t Vehicle, uint32_t Sequence, uint32_t Priority, uint32_t Packet_size, const char* DataRate);



  /**
   * Receive function
   * \param dev the device
   * \param pkt the packet
   * \param mode the mode
   * \param sender the sender address
   * \returns true if successful
   */
  bool Receive (Ptr<NetDevice> dev, Ptr<const Packet> pkt, uint16_t mode, const Address &sender);


  /// Create WAVE nodes function
  void CreateWaveNodes (void);


  NodeContainer nodes; ///< the nodes
  NetDeviceContainer devices; ///< the devices
};


void
WaveNetDeviceExample::CreateWaveNodes (void)
{

  //creating sender and receiver 
  nodes = NodeContainer ();
  nodes.Create (3); // the number of devices is 3 

  //installing mobility on the nodes since wifi nodes needs mobility to be installed
  MobilityHelper mobility;

 mobility.SetMobilityModel ("ns3::RandomWalk2dMobilityModel",
                             "Speed", StringValue ("ns3::UniformRandomVariable[Min=8|Max=13]"),   // Speed is between 8 to 13 m/s
                             "Bounds", RectangleValue (Rectangle (0, 60, 0, 100))); // Recangle Bounds = 0<X<60 and 0<Y<100

  mobility.SetPositionAllocator ("ns3::RandomRectanglePositionAllocator",
                                 "X", StringValue ("ns3::UniformRandomVariable[Min=0|Max=60]"), // Set the intial position of Vehicles Randomly
                                 "Y", StringValue ("ns3::UniformRandomVariable[Min=0|Max=100]") // Set the intial position of Vehicles Randomly
                                );
  mobility.Install (nodes);


  YansWifiChannelHelper waveChannel = YansWifiChannelHelper::Default ();
  YansWavePhyHelper wavePhy =  YansWavePhyHelper::Default ();
  wavePhy.SetChannel (waveChannel.Create ());
  wavePhy.SetPcapDataLinkType (WifiPhyHelper::DLT_IEEE802_11); //Set the data link type of PCAP traces to be used. This function has to be called before EnablePcap(), so that the header of the pcap file can be written correctly. Here we set it to IEEE 802.11 Wireless LAN headers 
  QosWaveMacHelper waveMac = QosWaveMacHelper::Default ();
  WaveHelper waveHelper = WaveHelper::Default ();
  devices = waveHelper.Install (wavePhy, waveMac, nodes);

  for (uint32_t i = 0; i != devices.GetN (); ++i)
    {
      Ptr<WaveNetDevice> device = DynamicCast<WaveNetDevice> (devices.Get (i)); //we use DynamicCast to use a subclass of a base class
      device->SetReceiveCallback (MakeCallback (&WaveNetDeviceExample::Receive, this));  //Set the callback to be used to notify higher layers when a packet has been received.

    }

  // Tracing
  wavePhy.EnablePcap ("wave-simple-device", devices);
}

bool
WaveNetDeviceExample::Receive (Ptr<NetDevice> dev, Ptr<const Packet> pkt, uint16_t mode, const Address &sender)
{
  SeqTsHeader seqTs;
  pkt->PeekHeader (seqTs);
  std::cout << "receive a packet: " << std::endl
            << "  reciever = " << dev->GetAddress() << "," << std::endl
            << "  sender = " << sender << "," << std::endl
            << "  sequence = " << seqTs.GetSeq () << "," << std::endl
            << "  sendTime = " << seqTs.GetTs ().As (Time::S) << "," << std::endl
            << "  recvTime = " << Now ().As (Time::S) << "," << std::endl
            << "  protocol = 0x" << std::hex << mode << std::dec  << std::endl;
  return true;
}

void
WaveNetDeviceExample::SendOneWsmpPacket (uint32_t Channel, uint32_t Vehicle ,uint32_t Sequence, uint32_t Priority, uint32_t Packet_size,const char* DataRate)
{
  Ptr<WaveNetDevice>  sender1 = DynamicCast<WaveNetDevice> (devices.Get (Vehicle));

  const static uint32_t WSMP_PROT_NUMBER = 0x88DC;   //WSM packets received from the lower layers with an Ethernet Type of 0x88DC are delivered to the WSM protocol. 
  Mac48Address bssWildcard = Mac48Address::GetBroadcast ();
  TxInfo txInfo = TxInfo (Channel, Priority, WifiMode(DataRate)); // The transmission power level is 10.
  Ptr<Packet> p  = Create<Packet> (Packet_size);   //Create a packet with a zero-filled payload of size 500.
  SeqTsHeader seqTs;       //SeqTsHeader: Packet header to carry sequence number and timestamp
  seqTs.SetSeq (Sequence);      //Set the sequence number in the header
  p->AddHeader (seqTs);    //Add header to a packet
  sender1->SendX  (p, bssWildcard, WSMP_PROT_NUMBER, txInfo);
}


void
WaveNetDeviceExample::SendWsmpExample ()
{
  CreateWaveNodes ();
  const SchInfo schInfo = SchInfo (SCH1, false, EXTENDED_ALTERNATING);   //alternating access to CCH and SCH
  // An important point is that the receiver should also be assigned channel
  // access for the same channel to receive packets.
  Ptr<WaveNetDevice>  sender1 = DynamicCast<WaveNetDevice> (devices.Get (0));
  int i = -1;

uint32_t time=1;
  while(time<=20) // Simulation time is 20 seconds
  {
    if(time==1)
    {
      Simulator::Schedule (Seconds (time-1), &WaveNetDevice::StartSch, sender1, schInfo);
      Simulator::Schedule (Seconds (time), &WaveNetDeviceExample::SendOneWsmpPacket,  this, CCH,0, 1,7,500,"OfdmRate12MbpsBW10MHz");
    }

    else if(time%5==0)
    {
      i+=2;  
      for (uint32_t Vehicle=0 ; Vehicle<=2 ; Vehicle++){  // Vehicles 1,2 and 3
        Ptr<WaveNetDevice>  sender2 = DynamicCast<WaveNetDevice> (devices.Get (Vehicle));
        Simulator::Schedule (Seconds (time-1), &WaveNetDevice::StartSch, sender2, schInfo);

        if (Vehicle == 0){
           Simulator::Schedule (Seconds (time), &WaveNetDeviceExample::SendOneWsmpPacket,  this, SCH1,0, time-Vehicle-i,0,1000,"OfdmRate27MbpsBW10MHz");
           continue;
        }
        if (Vehicle == 1){
          Simulator::Schedule (Seconds (time), &WaveNetDeviceExample::SendOneWsmpPacket,  this, SCH1,1, time-Vehicle-i,5,1000,"OfdmRate9MbpsBW10MHz");
          continue;
        }
        if (Vehicle == 2){
          Simulator::Schedule (Seconds (time), &WaveNetDeviceExample::SendOneWsmpPacket,  this, SCH1,2, time-Vehicle-i,7,1000,"OfdmRate6MbpsBW10MHz");
          continue;
        }
        }
      }
    time++;
    }
  Simulator::Schedule (Seconds (20.1), &WaveNetDevice::StopSch, sender1, SCH1);
  Simulator::Stop (Seconds (20.1));  
  Simulator::Run ();
  Simulator::Destroy ();
  }


int main (int argc, char *argv[])
{
  CommandLine cmd (__FILE__);
  cmd.Parse (argc, argv);
  
  WaveNetDeviceExample example;
  std::cout << "Group 23 ->>> run WAVE WSMP routing service case:" << std::endl;
  example.SendWsmpExample ();   //WAVE systems use a highly efficient messaging protocol known as WAVE Short Message Protocol (WSMP)

  return 0;
}