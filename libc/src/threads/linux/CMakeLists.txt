add_gen_header(
  thread_start_args_h
  DEF_FILE thread_start_args.h.def
  GEN_HDR thread_start_args.h
  PARAMS
    thread_start_args=${LIBC_TARGET_ARCHITECTURE}/thread_start_args.h.in
  DATA_FILES
    ${LIBC_TARGET_ARCHITECTURE}/thread_start_args.h.in
)

add_entrypoint_object(
  call_once
  SRCS
    call_once.cpp
  HDRS
    ../call_once.h
  DEPENDS
    .threads_utils
    libc.config.linux.linux_syscall_h
    libc.include.sys_syscall
    libc.include.threads
)

add_header_library(
  threads_utils
  HDRS
    Futex.h
    Mutex.h
    Thread.h
  DEPENDS
    .thread_start_args_h
    libc.config.linux.linux_syscall_h
    libc.include.sys_syscall
    libc.include.threads
)

add_entrypoint_object(
  thrd_create
  SRCS
    thrd_create.cpp
  HDRS
    ../thrd_create.h
  DEPENDS
    .threads_utils
    libc.config.linux.linux_syscall_h
    libc.include.errno
    libc.include.sys_syscall
    libc.include.threads
    libc.src.__support.common
    libc.src.errno.__errno_location
    libc.src.sys.mman.mmap
  COMPILE_OPTIONS
    -O3
    -fno-omit-frame-pointer # This allows us to sniff out the thread args from
                            # the new thread's stack reliably.
)

add_entrypoint_object(
  thrd_join
  SRCS
    thrd_join.cpp
  HDRS
    ../thrd_join.h
  DEPENDS
    .threads_utils
    libc.config.linux.linux_syscall_h
    libc.include.sys_syscall
    libc.include.threads
    libc.src.sys.mman.munmap
    libc.src.__support.common
)

add_entrypoint_object(
  mtx_init
  SRCS
    mtx_init.cpp
  HDRS
    ../mtx_init.h
  DEPENDS
    .threads_utils
    libc.include.threads
)

add_entrypoint_object(
  mtx_destroy
  SRCS
    mtx_destroy.cpp
  HDRS
    ../mtx_destroy.h
  DEPENDS
    .threads_utils
    libc.include.threads
)

add_entrypoint_object(
  mtx_lock
  SRCS
    mtx_lock.cpp
  HDRS
    ../mtx_lock.h
  DEPENDS
    .threads_utils
    libc.include.threads
)

add_entrypoint_object(
  mtx_unlock
  SRCS
    mtx_unlock.cpp
  HDRS
    ../mtx_unlock.h
  DEPENDS
    .threads_utils
    libc.include.threads
)

add_entrypoint_object(
  cnd_init
  SRCS
    cnd_init.cpp
  HDRS
    ../cnd_init.h
  DEPENDS
    .threads_utils
    libc.include.threads
)

add_entrypoint_object(
  cnd_destroy
  SRCS
    cnd_destroy.cpp
  HDRS
    ../cnd_destroy.h
  DEPENDS
    .threads_utils
    libc.include.threads
)

add_entrypoint_object(
  cnd_wait
  SRCS
    cnd_wait.cpp
  HDRS
    ../cnd_wait.h
  DEPENDS
    .threads_utils
    libc.include.threads
)

add_entrypoint_object(
  cnd_signal
  SRCS
    cnd_signal.cpp
  HDRS
    ../cnd_signal.h
  DEPENDS
    .threads_utils
    libc.include.threads
)

add_entrypoint_object(
  cnd_broadcast
  SRCS
    cnd_broadcast.cpp
  HDRS
    ../cnd_broadcast.h
  DEPENDS
    .threads_utils
    libc.include.threads
)
