"""
IOC 丰富化模块 - 从 OTX 获取 IOC 的详细信息
功能说明:
- 支持 IP、域名、主机名、URL 四种类型的 IOC
- 通过 OTX API 查询每个 IOC 的详细信息
- 提取关键威胁情报字段(地理位置、DNS 记录、服务器信息等)
- 统一返回标准化的数据结构
"""

from urllib.parse import urlparse
import OTXv2
from IndicatorTypes import IPv4, DOMAIN, URL  # OTXv2 包的一部分
from build_dataset.utils import infer_ioc_type


# ============================================================================
# 辅助函数
# ============================================================================

def sanitize(s):
    """
    清理字符串中的花括号

    问题: OTX API 使用 .format() 方法处理字符串
    某些 URL 中包含 {} 字符会导致格式化错误

    解决: 将 { 替换为 {{，将 } 替换为 }}，使其成为转义字符

    示例:
        'http://example.com/{path}'  →  'http://example.com/{{path}}'
    """
    return s.replace('{', '{{').replace('}', '}}')


# ============================================================================
# URL 类型 IOC 丰富化
# ============================================================================

def enrich_url(otx, url):
    """
    丰富化 URL 类型的 IOC

    查询信息:
    1. URL 的基本信息(主机名、地理位置)
    2. 网页服务器信息(IP、文件类型、HTTP 状态码)
    3. HTTP 响应头信息(Server、Content-Type、缓存策略等)
    4. 网页提取的元数据(标题、关键字等)

    参数:
        otx: OTX API 客户端
        url: 要查询的 URL 字符串

    返回:
        dict: 包含 URL 详细信息的字典
    """
    # 初始化基础字典，包含 URL 和类型
    base_dict = {
        'ioc': url,
        'type': 'URL',
        'hostname': urlparse(url).netloc  # 从 URL 提取主机名
    }

    try:
        # 调用 OTX API 获取 URL 的完整详细信息
        # 需要清理 URL 中的花括号以避免格式化错误
        details = otx.get_indicator_details_full(URL, sanitize(url))
    except (OTXv2.NotFound, OTXv2.BadRequest, OTXv2.RetryError):
        # 如果查询失败(URL 未找到或 API 错误)，返回基础信息
        return base_dict

    # 提取 WHOIS/地理位置信息
    whois = {k: v for k, v in details['general'].items() if k in [
        'net_loc',  # 网络位置
        'city',  # 城市
        'region',  # 地区
        'latitude',  # 纬度
        'longitude',  # 经度
        'country_code'  # 国家代码
    ]}
    base_dict.update(whois)

    # 获取 URL 列表详情(服务器信息)
    # 注意: OTX API 结构比较混乱，需要层层提取
    server_deets = details['url_list']['url_list']
    if server_deets:
        # 通常只有一个元素，取第一个的 result
        server_deets = server_deets[0]['result']
    else:
        # 如果没有服务器详情，返回已有信息
        return base_dict

    # API 有时会返回 [None]，需要额外检查
    if server_deets is None:
        return base_dict

    # ========== 提取网页信息 ==========
    urlworker = server_deets.get('urlworker', dict())
    # 提取关键的网页属性
    for k in ['ip', 'filetype', 'fileclass', 'http_code']:
        base_dict[k] = urlworker.get(k)

    # ========== 提取 HTTP 响应头信息 ==========
    if (resp := urlworker.get('http_response')):
        # 将所有键转换为大写，实现不区分大小写的访问
        new_resp = dict()
        for k in resp.keys():
            new_resp[k.upper()] = resp[k]
        resp = new_resp

        # 提取常见的 HTTP 响应头
        base_dict['server'] = resp.get('SERVER')  # 服务器软件
        base_dict['expires'] = resp.get('EXPIRES')  # 过期时间
        base_dict['cache-control'] = resp.get('CACHE-CONTROL')  # 缓存策略
        base_dict['encoding'] = resp.get('CONTENT-ENCODING')  # 内容编码
        base_dict['content-type'] = resp.get('CONTENT-TYPE')  # 内容类型

    # ========== 提取网页提取器信息 ==========
    # 包括网页标题、描述、关键字等元数据
    if (resp := server_deets.get('extractor')):
        # 给所有键加上 'extracted-' 前缀以示区分
        resp = {'extracted-' + k: v for k, v in resp.items()}
        base_dict.update(resp)

    return base_dict


# ============================================================================
# IP 地址类型 IOC 丰富化
# ============================================================================

def enrich_ip(otx, ip):
    """
    丰富化 IP 地址类型的 IOC (支持 IPv4 和 IPv6)

    查询信息:
    1. IP 的基本信息(地理位置、ASN)
    2. 被动 DNS 记录(该 IP 解析过哪些域名)
    3. DNS 记录的时间范围和类型

    参数:
        otx: OTX API 客户端
        ip: 要查询的 IP 地址字符串

    返回:
        dict: 包含 IP 详细信息的字典，包括解析记录列表
    """
    # 初始化基础字典
    base_dict = {
        'ioc': ip,
        'type': 'IP',
        'resolves_to': []  # 存储 DNS 解析记录
    }

    try:
        # 获取 IP 的基本信息(地理位置、ASN 等)
        details = otx.get_indicator_details_by_section(
            IPv4, ip, section='general'
        )

        # 提取 WHOIS/地理位置信息
        # 注意: 这些字段主要出现在 IPv6 查询中
        whois = {k: v for k, v in details.items() if k in [
            'net_loc',  # 网络位置
            'city',  # 城市
            'region',  # 地区
            'latitude',  # 纬度
            'longitude',  # 经度
            'country_code',  # 国家代码
            'asn'  # 自治系统编号
        ]}
        base_dict.update(whois)

    except (OTXv2.NotFound, OTXv2.BadRequest, OTXv2.RetryError):
        # 基本信息查询失败不影响后续流程
        pass

    try:
        # 获取被动 DNS 记录 - 这是最重要的信息
        # 被动 DNS 记录显示该 IP 曾经解析过哪些域名
        # 可以用于追踪攻击基础设施
        details = otx.get_indicator_details_by_section(
            IPv4, ip, section='passive_dns'
        )
    except (OTXv2.NotFound, OTXv2.BadRequest, OTXv2.RetryError):
        # 如果没有被动 DNS 记录，返回基础信息
        return base_dict

    # 提取被动 DNS 记录列表
    details = details['passive_dns']
    if len(details) == 0:
        return base_dict

    # 从第一条记录中提取 ASN(如果之前没有获取到)
    base_dict['asn'] = details[0]['asn']

    # 遍历所有 DNS 解析记录
    for resolution in details:
        base_dict['resolves_to'].append({
            'host': resolution.get('hostname'),  # 域名
            'record_type': resolution.get('record_type'),  # DNS 记录类型(A/AAAA/CNAME)
            'first_seen': resolution.get('first'),  # 首次发现时间
            'last_seen': resolution.get('last')  # 最后发现时间
        })

    return base_dict


# ============================================================================
# 域名类型 IOC 丰富化
# ============================================================================

def enrich_domain(otx, domain):
    """
    丰富化域名类型的 IOC

    查询信息:
    1. 域名的 DNS 记录(解析到哪些 IP)
    2. DNS 记录的时间范围和类型
    3. 相关的 ASN 信息

    参数:
        otx: OTX API 客户端
        domain: 要查询的域名字符串

    返回:
        dict: 包含域名详细信息的字典，包括 DNS 记录列表
    """
    # 初始化基础字典
    base_dict = {
        'ioc': domain,
        'type': 'domain',
        'dns_records': []  # 存储 DNS 记录
    }

    try:
        # 获取被动 DNS 记录
        # 显示该域名曾经解析到哪些 IP 地址
        details = otx.get_indicator_details_by_section(
            DOMAIN, domain, section='passive_dns'
        )
    except (OTXv2.NotFound, OTXv2.BadRequest, OTXv2.RetryError):
        # 如果没有 DNS 记录，返回基础信息
        return base_dict

    # 遍历所有被动 DNS 记录
    for record in details['passive_dns']:
        base_dict['dns_records'].append({
            k: record.get(k) for k in [
                'address',  # IP 地址
                'first',  # 首次发现时间
                'last',  # 最后发现时间
                'record_type',  # DNS 记录类型(A/AAAA/CNAME/MX)
                'asn'  # 自治系统编号
            ]
        })

    return base_dict


# ============================================================================
# 主机名类型 IOC 丰富化
# ============================================================================

def enrich_host(otx, host):
    """
    丰富化主机名类型的 IOC

    功能: 与域名处理相同，只是类型标记为 'hostname'
    主机名和域名在技术上类似，都需要 DNS 解析

    参数:
        otx: OTX API 客户端
        host: 要查询的主机名字符串

    返回:
        dict: 包含主机名详细信息的字典
    """
    # 复用域名丰富化函数
    ret_dict = enrich_domain(otx, host)
    # 修改类型标记为 hostname
    ret_dict['type'] = 'hostname'
    return ret_dict


# ============================================================================
# 统一入口函数
# ============================================================================

def enrich(otx, ioc, ioc_type):
    """
    IOC 丰富化的统一入口

    根据 IOC 类型自动调用相应的丰富化函数

    参数:
        otx: OTX API 客户端
        ioc: IOC 字符串(IP/域名/URL)
        ioc_type: IOC 类型标识

    返回:
        dict: 丰富化后的 IOC 信息字典

    异常:
        TypeError: 如果遇到不支持的 IOC 类型
    """
    # 根据类型路由到对应的处理函数
    if ioc_type.lower() == 'domain':
        return enrich_domain(otx, ioc)

    elif ioc_type in ['IPv4', 'IPv6', 'IP']:
        return enrich_ip(otx, ioc)

    elif ioc_type == 'URL':
        return enrich_url(otx, ioc)

    elif ioc_type.lower() == 'hostname':
        return enrich_host(otx, ioc)

    else:
        # 不支持的 IOC 类型
        raise TypeError("I don't know how to enrich %s" % ioc_type)


# ============================================================================
# 面向对象封装(可选)
# ============================================================================

class EnrichOTX():
    """
    IOC 丰富化类(面向对象封装)

    说明: 原作者更偏好函数式编程，所以主要逻辑在上面的函数中
    这个类只是一个封装包装器，方便需要使用对象的场景

    使用示例:
        enricher = EnrichOTX(api_key)
        result = enricher.enrich('8.8.8.8', 'IPv4')
        result = enricher.enrich('google.com')  # 自动推断类型
    """

    def __init__(self, api_key):
        """
        初始化 OTX 客户端

        参数:
            api_key: OTX API 密钥
        """
        self.otx = OTXv2.OTXv2(api_key)

    def enrich_ip(self, ip):
        """丰富化 IP 地址"""
        return enrich_ip(self.otx, ip)

    def enrich_domain(self, domain):
        """丰富化域名"""
        return enrich_domain(self.otx, domain)

    def enrich_host(self, host):
        """丰富化主机名"""
        return enrich_host(self.otx, host)

    def enrich_url(self, url):
        """丰富化 URL"""
        return enrich_url(self.otx, url)

    def enrich(self, ioc, ioc_type=None):
        """
        通用丰富化方法(支持自动类型推断)

        参数:
            ioc: IOC 字符串
            ioc_type: IOC 类型(可选)

        返回:
            dict: 丰富化后的信息

        注意:
            如果不提供 ioc_type，会尝试自动推断
            自动推断有风险，建议提供明确的类型
        """
        if ioc_type is None:
            # 尝试自动推断 IOC 类型
            # 注意: 如果推断失败会抛出 TypeError
            # 更安全的做法是明确提供类型参数
            ioc_type = infer_ioc_type(ioc)

        # 调用统一入口函数
        return enrich(self.otx, ioc, ioc_type)


# ============================================================================
# 使用示例
# ============================================================================
"""
# 函数式调用
from OTXv2 import OTXv2

otx = OTXv2(api_key)

# 丰富化 IP
ip_info = enrich_ip(otx, '8.8.8.8')
print(ip_info['resolves_to'])  # 查看 DNS 记录

# 丰富化域名
domain_info = enrich_domain(otx, 'example.com')
print(domain_info['dns_records'])  # 查看 DNS 记录

# 丰富化 URL
url_info = enrich_url(otx, 'http://malicious.com/payload')
print(url_info['server'])  # 查看服务器信息

# 面向对象调用
enricher = EnrichOTX(api_key)
result = enricher.enrich('8.8.8.8', 'IPv4')
result = enricher.enrich('example.com')  # 自动推断类型
"""