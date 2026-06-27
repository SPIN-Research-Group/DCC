#ifndef     RAMULATOR_BASE_REQUEST_H
#define     RAMULATOR_BASE_REQUEST_H

#include <vector>
#include <list>
#include <string>

#include "base/base.h"

namespace Ramulator {

struct Request { 
  Addr_t    addr = -1;
  AddrVec_t addr_vec {};

  // Basic request id convention
  // 0 = Read, 1 = Write. The device spec defines all others
  struct Type {
    enum : int {
      // DRAM requests
      Read = 0, 
      Write = 1,

      // PIM requests
      PIM_MAC_AB = 4,
      PIM_MAC_SB = 5,
      PIM_MAC_PB = 6,
      PIM_WR_GB = 7,
      PIM_MV_SB = 8,
      PIM_MV_GB = 9,
      PIM_SFM = 10,
      PIM_SET_MODEL = 11,
      PIM_SET_HEAD = 12,
      PIM_BARRIER = 13,
      PIM_DUMMY  = 14,
      PIM_LD_OP1 = 15,
      PIM_LD_OP2 = 16,
      PIM_LD_OP3 = 17,
      PIM_MUL_OP1  = 18,
      PIM_MUL_OP2  = 19,
      PIM_MUL_OP3  = 20,
      PIM_ADD_OP1  = 21,
      PIM_ADD_OP2  = 22,
      PIM_ADD_OP3  = 23,
      PIM_MAC_OP1  = 24,
      PIM_MAC_OP2  = 25,
      PIM_MAC_OP3  = 26,
      PIM_WB_ACC = 27,
      PIM_WB_RES = 28,
      PIM_ACC_RESET = 29,
      PIM_RELU = 30
    };
  };

  int type_id = -1;    // An identifier for the type of the request
  int source_id = -1;  // An identifier for where the request is coming from (e.g., which core)

  int command = -1;          // The command that need to be issued to progress the request
  int final_command = -1;    // The final command that is needed to finish the request

  Clk_t arrive = -1;   // Clock cycle when the request arrive at the memory controller
  Clk_t depart = -1;   // Clock cycle when the request depart the memory controller

  std::function<void(Request&)> callback;

  Request(Addr_t addr, int type);
  Request(AddrVec_t addr_vec, int type);
  Request(Addr_t addr, int type, int source_id, std::function<void(Request&)> callback);
};


struct ReqBuffer {
  std::list<Request> buffer;
  size_t max_size = 32;


  using iterator = std::list<Request>::iterator;
  iterator begin() { return buffer.begin(); };
  iterator end() { return buffer.end(); };


  size_t size() const { return buffer.size(); }

  bool enqueue(const Request& request) {
    if (buffer.size() <= max_size) {
      buffer.push_back(request);
      return true;
    } else {
      return false;
    }
  }

  void remove(iterator it) {
    buffer.erase(it);
  }
};

}        // namespace Ramulator


#endif   // RAMULATOR_BASE_REQUEST_H