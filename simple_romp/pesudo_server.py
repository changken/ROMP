import socket

# socket config
HOST = '127.0.0.1'
PORT = 8000
server_addr = (HOST, PORT)

#s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.bind(server_addr)

print('server start at: %s:%s' % (HOST, PORT))
print('wait for connection...')

def main():
    global s

    while True:
        # 54 * 3 * 11 = 1944 => 有可能會超過3400
        indata, addr = s.recvfrom(4096)
        print('recvfrom ' + str(addr) + ': ' + indata.decode())

        #outdata = 'echo ' + indata.decode()
        #outdata = 'recevied data!'
        #s.sendto(outdata.encode(), addr)

if __name__ == "__main__":
    main()