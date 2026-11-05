from playwright.async_api import async_playwright


class Browser:
    def __init__(self):
        self.playwright = None
        self.browser = None
        self.page = None

    async def start(self):
        if self.playwright:
            return

        self.playwright = await async_playwright().start()
        self.browser = await self.playwright.chromium.launch(
            headless=False
        )
        self.page = await self.browser.new_page()

    async def stop(self):
        if self.browser:
            await self.browser.close()

        if self.playwright:
            await self.playwright.stop()

        self.playwright = None
        self.browser = None
        self.page = None

    async def navigate(self, url: str):
        if not self.page:
            raise RuntimeError("Browser is not running")

        await self.page.goto(url, wait_until="domcontentloaded")
        return await self.inspect()

    async def inspect(self):
        if not self.page:
            raise RuntimeError("Browser is not running")

        return await self.page.evaluate("""() => ({
            url: location.href,
            title: document.title,
            text: document.body.innerText.slice(0, 12000),

            buttons: [...document.querySelectorAll('button')]
                .slice(0, 50)
                .map(b => ({
                    text: b.innerText.trim(),
                    disabled: b.disabled
                })),

            inputs: [...document.querySelectorAll('input')]
                .slice(0, 50)
                .map(i => ({
                    type: i.type,
                    placeholder: i.placeholder,
                    name: i.name
                })),

            links: [...document.querySelectorAll('a')]
                .slice(0, 50)
                .map(a => ({
                    text: a.innerText.trim(),
                    href: a.href
                }))
        })""")

    async def get_page(self):
        if not self.page:
            raise RuntimeError("Browser is not running")

        return self.page


browser = Browser()