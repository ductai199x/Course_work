#include <stdio.h>
#include <unistd.h>
#include <signal.h>
#include <stdlib.h>
#include <errno.h>
#include <string.h>

#define DEBUG 1

void print_usage()
{
    printf(
        "Usage: ./signal [options] <pid>\n"
        "Options:\n"
        "\t-s <signal>\tSends <signal> to <pid>\n"
        "\t-l\t\tLists all signal numbers with their names\n");
}

void list_all_sig()
{
    printf(
        "+--------+-----------+\n"
        "| Number | Name      |\n"
        "+--------+-----------|\n"
        "|    1   | SIGHUP    |\n"
        "|    2   | SIGINT    |\n"
        "|    3   | SIGQUIT   |\n"
        "|    4   | SIGILL    |\n"
        "|    5   | SIGTRAP   |\n"
        "|    6   | SIGABRT   |\n"
        "|    7   | SIGBUS    |\n"
        "|    8   | SIGFPE    |\n"
        "|    9   | SIGKILL   |\n"
        "|   10   | SIGUSR1   |\n"
        "|   11   | SIGSEGV   |\n"
        "|   12   | SIGUSR2   |\n"
        "|   13   | SIGPIPE   |\n"
        "|   14   | SIGALRM   |\n"
        "|   15   | SIGTERM   |\n"
        "|   16   | SIGSTKFLT |\n"
        "|   17   | SIGCHLD   |\n"
        "|   18   | SIGCONT   |\n"
        "|   19   | SIGSTOP   |\n"
        "|   20   | SIGTSTP   |\n"
        "|   21   | SIGTTIN   |\n"
        "|   22   | SIGTTOU   |\n"
        "|   23   | SIGURG    |\n"
        "|   24   | SIGXCPU   |\n"
        "|   25   | SIGXFSZ   |\n"
        "|   26   | SIGVTALRM |\n"
        "|   27   | SIGPROF   |\n"
        "|   28   | SIGWINCH  |\n"
        "|   29   | SIGIO     |\n"
        "|   30   | SIGPWR    |\n"
        "|   31   | SIGSYS    |\n"
        "+--------+-----------+\n"
        );
}

void send_signal(int signal, long unsigned int pid)
{
    if ( kill(pid, signal) == -1 ) {
        if ( errno == EPERM ) {
            fprintf(stderr, "PID %ld exists, but we can't send it signals\n", pid);
        }
        else if ( errno == EINVAL ) {
            fprintf(stderr, "Invalid signal %d\n", signal);
        }
        else if ( errno == ESRCH ) {
            fprintf(stderr, "PID %ld does not exist\n", pid);
        } 
        else {
            fprintf(stderr, "ERROR sending signal %d to pid %ld. ERRNO=%d\n", signal, pid, errno);
        }
    }
    else {
        if ( signal == 0 ) {
            printf("PID %ld exists and is able to receive signals\n", pid);
        }
        else {
            printf("Sent signal %d to pid %ld\n", signal, pid);
        }
    }

}

int main (int argc, char** argv)
{
    if ( argc < 2 ) {
        print_usage();
        exit(1);
    }

    long unsigned int pid;

    if ( argc == 2 ) {
        if ( !strcmp("-l", argv[1]) ) {
            // List all signals
            list_all_sig();
        }
        else if ( !strcmp("-s", argv[1]) ) {
            fprintf(stderr, "Missing pid\n");
        }
        else {
            // Send SIGTERM to pid
            pid = strtol(argv[1], (char**)NULL, 10);
            if ( pid != 0 ) {
                send_signal(15, pid);
            }
            else {
                fprintf(stderr, "Invalid option\n");
                print_usage();
            }
        }
        exit(1);
    }

    char* option = argv[1];
    int signal = strtol(argv[2], (char**)NULL, 10);
    
    if ( argc == 4 ) {
        if ( !strcmp("-s", option) ) {
            // Send specific signal to pid
            
            pid = strtol(argv[3], (char**)NULL, 10);
            send_signal(signal, pid); 
       }
        else {
            printf("Option doesn't exist.\n");
            print_usage();
        }
        exit(1);
    }

    if ( argc == 3 || argc > 4 ) {
        fprintf(stderr, "Invalid arguments\n");
        print_usage();
    }

}
