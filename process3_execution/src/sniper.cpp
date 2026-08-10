#include <iostream>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <cstdint>
#include <string>
#include <fstream>
#include <sstream>
#include <curl/curl.h>

#pragma pack(push, 1)
struct ExecutionSignal
{
    bool fire_order;
    double target_price;
    uint32_t target_qty;
    bool is_active;
};
#pragma pack(pop)

// Global variables to hold secure credentials
std::string dhan_client_id = "";
std::string dhan_access_token = "";
const std::string SECURITY_ID = "1333";

void load_env(const std::string &filepath)
{
    std::ifstream file(filepath);
    if (!file.is_open())
    {
        std::cerr << "[WARNING] Could not open .env file at " << filepath << std::endl;
        return;
    }

    std::string line;
    while (std::getline(file, line))
    {
        std::istringstream is_line(line);
        std::string key;
        if (std::getline(is_line, key, '='))
        {
            std::string value;
            if (std::getline(is_line, value))
            {
                // Strip quotes if they exist in the .env file
                if (value.size() >= 2 && value.front() == '"' && value.back() == '"')
                {
                    value = value.substr(1, value.size() - 2);
                }
                if (key == "DHAN_CLIENT_ID")
                    dhan_client_id = value;
                if (key == "DHAN_ACCESS_TOKEN")
                    dhan_access_token = value;
            }
        }
    }
}

void execute_dhan_order(double price, uint32_t qty)
{
    CURL *curl;
    CURLcode res;
    curl = curl_easy_init();

    if (curl)
    {
        curl_easy_setopt(curl, CURLOPT_URL, "https://api.dhan.co/v2/orders");

        struct curl_slist *headers = NULL;
        headers = curl_slist_append(headers, "Content-Type: application/json");
        std::string auth_header = "access-token: " + dhan_access_token;
        headers = curl_slist_append(headers, auth_header.c_str());
        curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);

        std::string json_payload = "{"
                                   "\"dhanClientId\":\"" +
                                   dhan_client_id + "\","
                                                    "\"correlationId\":\"yolo_sniper_01\","
                                                    "\"transactionType\":\"BUY\","
                                                    "\"exchangeSegment\":\"NSE_EQ\","
                                                    "\"productType\":\"INTRADAY\","
                                                    "\"orderType\":\"LIMIT\","
                                                    "\"validity\":\"DAY\","
                                                    "\"securityId\":\"" +
                                   SECURITY_ID + "\","
                                                 "\"quantity\":" +
                                   std::to_string(qty) + ","
                                                         "\"disclosedQuantity\":0,"
                                                         "\"price\":" +
                                   std::to_string(price) + ","
                                                           "\"triggerPrice\":0,"
                                                           "\"afterMarketOrder\":false"
                                                           "}";

        curl_easy_setopt(curl, CURLOPT_POSTFIELDS, json_payload.c_str());

        res = curl_easy_perform(curl);
        if (res != CURLE_OK)
        {
            std::cerr << "\n[ERROR] Dhan API execution failed: " << curl_easy_strerror(res) << std::endl;
        }
        else
        {
            std::cout << "\n[SUCCESS] Order payload sent to Dhan API!" << std::endl;
        }

        curl_slist_free_all(headers);
        curl_easy_cleanup(curl);
    }
}

int main()
{
    // Load credentials from the root .env file
    load_env("../.env");
    if (dhan_client_id.empty() || dhan_access_token.empty())
    {
        std::cerr << "[FATAL] Failed to load credentials from .env" << std::endl;
        return 1;
    }

    curl_global_init(CURL_GLOBAL_ALL);

    const char *shm_name = "/dhan_sniper_bridge";
    int shm_fd = shm_open(shm_name, O_RDONLY, 0666);
    if (shm_fd == -1)
    {
        std::cerr << "Failed to find Shared Memory. Is Process 2 running?" << std::endl;
        return 1;
    }

    void *ptr = mmap(0, sizeof(ExecutionSignal), PROT_READ, MAP_SHARED, shm_fd, 0);
    if (ptr == MAP_FAILED)
    {
        std::cerr << "Memory map failed." << std::endl;
        close(shm_fd);
        return 1;
    }

    volatile ExecutionSignal *signal = static_cast<volatile ExecutionSignal *>(ptr);
    std::cout << "C++ Sniper armed and watching memory block: " << shm_name << std::endl;

    while (true)
    {
        if (signal->fire_order)
        {
            std::cout << "\n[EXECUTE!] Firing " << signal->target_qty
                      << " shares at price: " << signal->target_price << std::endl;

            execute_dhan_order(signal->target_price, signal->target_qty);

            sleep(1);
        }

        if (!signal->is_active)
        {
            std::cout << "Process 2 disconnected. Shutting down sniper." << std::endl;
            break;
        }
    }

    munmap(ptr, sizeof(ExecutionSignal));
    close(shm_fd);
    curl_global_cleanup();
    return 0;
}